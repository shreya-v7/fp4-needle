"""Character-level GPT (decoder-only Transformer) trained with FP4 matmuls.

Data: text8 (first 100M characters of Wikipedia, 27 symbols), or any text file.
Every Transformer linear layer and the output layer follow the QuantConfig; the
character embedding stays FP32. Metric: bits per character (loss / ln 2).

    python3 apps/fp4_gpt_experiment.py --data-file data/text8 --size small --steps 2000 --device cuda
"""
import argparse
import csv
import math
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.append(os.path.dirname(__file__))

import numpy as np

import needle as ndl
import needle.nn as nn
from needle.fp4.config import QuantConfig
from needle.fp4 import monitor
from fp4_common import save_checkpoint

SIZES = {"tiny": (64, 2, 4), "small": (128, 4, 4), "medium": (256, 6, 8)}   # d_model, layers, heads

CONFIGS = {
    "grid": [QuantConfig(), QuantConfig(fprop=True), QuantConfig(fprop=True, dgrad=True),
             QuantConfig(fprop=True, dgrad=True, wgrad=True),
             QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True)],
    "ablation": [QuantConfig(dgrad=True), QuantConfig(dgrad=True, stochastic=True),
                 QuantConfig(wgrad=True), QuantConfig(wgrad=True, stochastic=True)],
    "main": [QuantConfig(), QuantConfig(fprop=True, dgrad=True, wgrad=True),
             QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True)],
}
CONFIGS["all"] = CONFIGS["grid"] + CONFIGS["ablation"]
CSV_FIELDS = ["config", "seed", "split", "epoch", "step", "loss", "accuracy", "elapsed_s"]


class MiniGPT(nn.Module):
    def __init__(self, vocab, d_model, layers, heads, seq_len, config, device):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab, d_model, device=device)
        self.transformer = nn.Transformer(d_model, 4 * d_model, layers, num_head=heads,
                                          dim_head=d_model // heads, causal=True,
                                          sequence_len=seq_len, device=device, config=config)
        self.head = nn.QuantizedLinear(d_model, vocab, device=device, config=config)

    def forward(self, x):
        seq_len, batch = x.shape
        h, _ = self.transformer(self.embedding(x))
        return self.head(h.reshape((seq_len * batch, self.d_model)))


def load_text(path, max_chars):
    with open(path, encoding="utf-8") as f:
        text = f.read(max_chars) if max_chars else f.read()
    chars = sorted(set(text))
    lookup = {c: i for i, c in enumerate(chars)}
    data = np.fromiter((lookup[c] for c in text), dtype=np.float32, count=len(text))
    n = len(data)
    return chars, data[: int(0.9 * n)], data[int(0.9 * n): int(0.95 * n)], data[int(0.95 * n):]


def num_params(model):
    return sum(int(np.prod(p.shape)) for p in model.parameters())


def evaluate(model, data, args, device, max_batches):
    model.eval()
    loss_fn = nn.SoftmaxLoss()
    total, correct, n = 0.0, 0, 0
    for k, i in enumerate(range(0, data.shape[0] - 1, args.seq_len)):
        if k >= max_batches:
            break
        X, y = ndl.data.get_batch(data, i, args.seq_len, device=device, dtype="float32")
        logits = model(X)
        total += float(loss_fn(logits, y).numpy()) * y.shape[0]
        correct += int((logits.numpy().argmax(1) == y.numpy()).sum())
        n += y.shape[0]
    model.train()
    return total / n, correct / n


def ptq_target(meta, device):
    chars, _, val, test = load_text(meta["data_file"], meta["max_chars"])
    d, layers, heads = SIZES[meta["size"]]
    model = MiniGPT(len(chars), d, layers, heads, meta["seq_len"], QuantConfig(), device)
    test = ndl.data.batchify(test, meta["batch_size"], device=device, dtype="float32")
    args = argparse.Namespace(seq_len=meta["seq_len"])
    return model, lambda m: {"test_bpc": evaluate(m, test, args, device, meta["eval_batches"])[0] / math.log(2)}


def train_one(config, args, data, device, writer, gnr_writer):
    chars, train, val, _ = data
    np.random.seed(args.seed)
    d, layers, heads = SIZES[args.size]
    model = MiniGPT(len(chars), d, layers, heads, args.seq_len, config, device)
    name = config.name + (f"+switch{args.switch_step}" if args.switch_step else "")
    opt = ndl.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SoftmaxLoss()
    train_b = ndl.data.batchify(train, args.batch_size, device=device, dtype="float32")
    val_b = ndl.data.batchify(val, args.batch_size, device=device, dtype="float32")
    start, starts = time.time(), list(range(0, train_b.shape[0] - 1, args.seq_len))

    def log(split, step, loss, acc):
        writer.writerow({"config": name, "seed": args.seed, "split": split, "epoch": step // args.eval_every,
                         "step": step, "loss": f"{loss:.6f}", "accuracy": f"{acc:.4f}",
                         "elapsed_s": f"{time.time() - start:.1f}"})

    for step in range(args.steps):
        X, y = ndl.data.get_batch(train_b, starts[step % len(starts)], args.seq_len, device=device, dtype="float32")
        if args.switch_step and step == args.switch_step:
            monitor.set_config(model, monitor.exact_backward)
        if gnr_writer and step % args.monitor_every == 0:
            r = monitor.grad_noise_ratio(model, lambda: loss_fn(model(X), y))
            gnr_writer.writerow({"config": name, "seed": args.seed, "step": step, "ratio": f"{r:.5g}"})
        opt.reset_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        value = float(loss.numpy())
        log("train", step + 1, value, float((logits.numpy().argmax(1) == y.numpy()).mean()))
        if not np.isfinite(value):
            print(f"  {name}: loss became {value} at step {step + 1}; stopping run")
            return
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vl, va = evaluate(model, val_b, args, device, args.eval_batches)
            log("test", step + 1, vl, va)
            print(f"  {name:24s} step {step + 1}: val bpc {vl / math.log(2):.3f}  "
                  f"({time.time() - start:.0f}s)", flush=True)
    if args.save_weights:
        save_checkpoint(model, args.save_weights.format(config=name, seed=args.seed, size=args.size),
                        {"runner": "gpt", "data_file": args.data_file, "max_chars": args.max_chars,
                         "size": args.size, "seq_len": args.seq_len, "batch_size": args.batch_size,
                         "eval_batches": args.eval_batches, "config": name})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--configs", choices=CONFIGS, default="main")
    p.add_argument("--only", nargs="+", metavar="NAME")
    p.add_argument("--device", choices=["cpu", "cuda", "cpu_numpy"], default="cpu")
    p.add_argument("--data-file", default="data/text8")
    p.add_argument("--max-chars", type=int, default=5_000_000, help="0 = whole file")
    p.add_argument("--size", choices=SIZES, default="small")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/gpt_seed0.csv")
    p.add_argument("--save-weights", default="", help="may use {config}, {seed}, {size}")
    p.add_argument("--monitor-every", type=int, default=0)
    p.add_argument("--switch-step", type=int, default=0)
    args = p.parse_args()

    device = getattr(ndl, args.device)()
    if not device.enabled():
        p.error(f"needle's {args.device} backend is not built (Colab: use a GPU runtime)")
    data = load_text(args.data_file, args.max_chars)
    configs = [c for c in CONFIGS[args.configs] if not args.only or c.name in args.only]
    d, layers, heads = SIZES[args.size]
    probe = MiniGPT(len(data[0]), d, layers, heads, args.seq_len, QuantConfig(), ndl.cpu())
    print(f"{args.data_file}: {len(data[1])} train chars, vocab {len(data[0])}; "
          f"size {args.size} = {num_params(probe) / 1e6:.2f}M params; configs {[c.name for c in configs]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gnr_file = open(args.out.replace(".csv", "_gnr.csv"), "w", newline="") if args.monitor_every else None
    gnr_writer = csv.DictWriter(gnr_file, fieldnames=["config", "seed", "step", "ratio"]) if gnr_file else None
    if gnr_writer:
        gnr_writer.writeheader()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for config in configs:
            train_one(config, args, data, device, writer, gnr_writer)
            f.flush()
            if gnr_file:
                gnr_file.flush()
    if gnr_file:
        gnr_file.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
