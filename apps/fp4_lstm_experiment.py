"""LSTM language model on Penn Treebank with FP4 matmuls (the TA's "LSTM with SR").

Same idea as apps/fp4_experiment.py (MNIST MLP), on a recurrent model: every
matmul of the LSTM (X @ W_ih and h @ W_hh, at every time step) and of the
output layer runs in FP4 according to a QuantConfig. The embedding is a table
lookup, not a real matmul, so it stays FP32.

Language models are scored by perplexity = exp(average cross-entropy loss):
"the model is as unsure as if it picked uniformly among this many words".
Lower is better. The CSV has the same columns as the MNIST runner (loss and
next-word accuracy), so apps/plot_fp4.py plots it with --perplexity.

    python3 apps/fp4_lstm_experiment.py --device cuda --epochs 3
    python3 apps/plot_fp4.py results/lstm_seed0.csv --out-dir results/lstm --perplexity \\
        --title "PTB LSTM language model"
"""
import argparse
import csv
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

import needle as ndl
import needle.nn as nn
from needle.fp4.config import QuantConfig

CONFIGS = {
    "grid": [QuantConfig(),
             QuantConfig(fprop=True),
             QuantConfig(fprop=True, dgrad=True),
             QuantConfig(fprop=True, dgrad=True, wgrad=True),
             QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True)],
    "ablation": [QuantConfig(dgrad=True), QuantConfig(dgrad=True, stochastic=True),
                 QuantConfig(wgrad=True), QuantConfig(wgrad=True, stochastic=True)],
}
CONFIGS["all"] = CONFIGS["grid"] + CONFIGS["ablation"]
CSV_FIELDS = ["config", "seed", "split", "epoch", "step", "loss", "accuracy", "elapsed_s"]


class FP4LanguageModel(nn.Module):
    """Embedding (FP32) -> LSTM (FP4 per config) -> linear output layer (FP4 per config).

    The same architecture as LanguageModel in apps/models.py with seq_model='lstm'."""

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, config, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, device=device)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device,
                            config=config)
        self.linear = nn.QuantizedLinear(hidden_size, vocab_size, device=device, config=config)

    def forward(self, x, h=None):
        seq_len, batch_size = x.shape
        out, h = self.lstm(self.embedding(x), h)
        return self.linear(out.reshape((seq_len * batch_size, self.hidden_size))), h


def run_epoch(model, data, args, device, opt=None, on_step=None):
    """One pass over `data` (nbatch, batch_size) with truncated backprop through time.
    Returns (average loss, next-word accuracy)."""
    model.train() if opt is not None else model.eval()
    loss_fn = nn.SoftmaxLoss()
    h, total_loss, total_correct, n = None, 0.0, 0, 0
    for i in range(0, data.shape[0] - 1, args.seq_len):
        X, y = ndl.data.get_batch(data, i, args.seq_len, device=device, dtype="float32")
        logits, h = model(X, h)
        loss = loss_fn(logits, y)
        h = tuple(t.detach() for t in h)        # stop gradients at the chunk boundary
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            if args.clip:
                opt.clip_grad_norm(args.clip)
            opt.step()
        loss_value = float(loss.numpy())
        correct = int((logits.numpy().argmax(axis=1) == y.numpy()).sum())
        total_loss += loss_value * y.shape[0]
        total_correct += correct
        n += y.shape[0]
        if on_step is not None:
            on_step(loss_value, correct / y.shape[0])
    return total_loss / n, total_correct / n


def train_one(config, corpus, args, device, writer):
    # Same seed for every config: same initial weights; the data order is fixed
    # (PTB is read in order), so runs differ only by FP4.
    np.random.seed(args.seed)
    vocab = len(corpus.dictionary)
    model = FP4LanguageModel(vocab, args.embedding, args.hidden, args.layers, config, device)
    opt = (ndl.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam"
           else ndl.optim.SGD(model.parameters(), lr=args.lr))
    train = ndl.data.batchify(corpus.train, args.batch_size, device=device, dtype="float32")
    test = ndl.data.batchify(corpus.test, args.batch_size, device=device, dtype="float32")
    start, step = time.time(), [0]

    def log(split, epoch, loss, acc):
        writer.writerow({"config": config.name, "seed": args.seed, "split": split, "epoch": epoch,
                         "step": step[0], "loss": f"{loss:.6f}", "accuracy": f"{acc:.4f}",
                         "elapsed_s": f"{time.time() - start:.1f}"})

    for epoch in range(1, args.epochs + 1):
        def on_step(loss, acc):
            step[0] += 1
            log("train", epoch, loss, acc)
        train_loss, _ = run_epoch(model, train, args, device, opt=opt, on_step=on_step)
        if not np.isfinite(train_loss):
            print(f"  {config.name}: loss became {train_loss}; stopping run")
            return
        test_loss, test_acc = run_epoch(model, test, args, device)
        log("test", epoch, test_loss, test_acc)
        print(f"  {config.name:24s} epoch {epoch}: train ppl {np.exp(train_loss):8.1f}  "
              f"test ppl {np.exp(test_loss):8.1f}  ({time.time() - start:.0f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", choices=CONFIGS, default="grid")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    parser.add_argument("--device", choices=["cpu", "cuda", "cpu_numpy"], default="cpu")
    parser.add_argument("--data-dir", default="data/ptb")
    parser.add_argument("--out", default="results/lstm_seed0.csv")
    parser.add_argument("--max-lines", type=int, default=2000,
                        help="lines of train.txt / test.txt to use (PTB has 42068 / 3761)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--clip", type=float, default=0.0, help="max gradient norm (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = getattr(ndl, args.device)()
    if not device.enabled():
        parser.error(f"needle's {args.device} backend is not built (Colab: use a GPU runtime)")
    corpus = ndl.data.Corpus(args.data_dir, max_lines=args.max_lines)
    configs = CONFIGS[args.configs]
    if args.only:
        configs = [c for c in configs if c.name in args.only]
    print(f"PTB: {len(corpus.train)} train / {len(corpus.test)} test tokens, "
          f"vocabulary {len(corpus.dictionary)}; training {[c.name for c in configs]} on {device}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for config in configs:
            train_one(config, corpus, args, device, writer)
            f.flush()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
