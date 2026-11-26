# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Stage 3 experiment runner: train an MLP on MNIST or CIFAR-10 with FP4 linear layers.

Each run trains the same MLP (same seed, same data order) with a different
QuantConfig and logs to one CSV file:
    * one row per training step:   split=train, the batch loss and accuracy
    * one row per epoch:           split=test,  loss and accuracy on the test set

Examples (from the repo root):
    # the 6 distinct configs of the project plan, short CPU run
    python3 apps/fp4_experiment.py --configs grid --device cpu --epochs 1 --train-subset 20000
    # every config incl. single-matmul ablations, full data, on a GPU
    python3 apps/fp4_experiment.py --configs all --device cuda --epochs 3
    # the same on CIFAR-10
    python3 apps/fp4_experiment.py --dataset cifar10 --configs all --device cuda --epochs 3
    # then plot
    python3 apps/plot_fp4.py results/mnist.csv

NumPy is used here only for data loading and metrics (argmax / accuracy),
like the homework's apps/simple_ml.py; all model math runs in needle.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.append(os.path.dirname(__file__))

import numpy as np

import needle as ndl
import needle.nn as nn
from needle.fp4.config import QuantConfig
from needle.fp4.quantize import seed_sr, set_sr_state, sr_state
from fp4_common import save_checkpoint

# ----------------------------------------------------------------------------
# Which configurations to train
# ----------------------------------------------------------------------------

# The project grid {FP32, Fprop, Fprop+Dgrad, Fprop+Dgrad+Wgrad} x {SR off, on}.
# SR only applies to gradient operands (see QuantConfig), so "FP32 + SR" and
# "Fprop + SR" would be identical to their SR-off runs; we train each once.
GRID = [
    QuantConfig(),
    QuantConfig(fprop=True),
    QuantConfig(fprop=True, dgrad=True),
    QuantConfig(fprop=True, dgrad=True, stochastic=True),
    QuantConfig(fprop=True, dgrad=True, wgrad=True),
    QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True),
]

# One backward matmul in FP4 at a time, to see which one hurts on its own.
ABLATION = [
    QuantConfig(dgrad=True),
    QuantConfig(dgrad=True, stochastic=True),
    QuantConfig(wgrad=True),
    QuantConfig(wgrad=True, stochastic=True),
]

# All three matmuls in FP4, with W quantized consistently for Fprop and Dgrad:
# one scale per 16x16 tile (NVIDIA) or SR re-quantization of the forward W (Thinking Machines).
WEIGHTS = [QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=sr, weights=w)
           for sr in (False, True) for w in ("2d", "sr")]

# Dgrad with only one operand quantized, to separate the gradient dY from the weight W.
ISOLATE = [QuantConfig(dgrad=True, dgrad_operands="dy"),
           QuantConfig(dgrad=True, dgrad_operands="dy", stochastic=True),
           QuantConfig(dgrad=True, dgrad_operands="w")]

# The extended study: the configurations that carry the result, plus the isolation runs.
EXTENDED = [QuantConfig(), QuantConfig(dgrad=True), QuantConfig(dgrad=True, stochastic=True)] + ISOLATE + [
    QuantConfig(fprop=True, dgrad=True, wgrad=True),
    QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True)]

# For the learning-rate and depth sweeps.
CORE = [QuantConfig(), QuantConfig(dgrad=True), QuantConfig(dgrad=True, stochastic=True)]

CONFIG_SETS = {"grid": GRID, "ablation": ABLATION, "weights": WEIGHTS, "isolate": ISOLATE,
               "extended": EXTENDED, "core": CORE, "all": GRID + ABLATION}

CSV_FIELDS = ["config", "seed", "split", "epoch", "step", "loss", "accuracy", "elapsed_s"]


# ----------------------------------------------------------------------------
# Model and data
# ----------------------------------------------------------------------------

def make_mlp(config, hidden, device, in_features=784, depth=2):
    """MLP with `depth` hidden layers of width `hidden` (in -> hidden x depth -> 10), every
    linear layer using `config`. depth=2 is the model used throughout the report."""
    layers = [nn.Flatten(), nn.QuantizedLinear(in_features, hidden, device=device, config=config), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.QuantizedLinear(hidden, hidden, device=device, config=config), nn.ReLU()]
    layers.append(nn.QuantizedLinear(hidden, 10, device=device, config=config))
    return nn.Sequential(*layers)


def load_mnist(data_dir, train_subset):
    train = ndl.data.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                                  os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test = ndl.data.MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                                 os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    if train_subset:  # use the first N training images for quick runs
        train.images, train.labels = train.images[:train_subset], train.labels[:train_subset]
    return train, test


def load_cifar10(data_dir, train_subset):
    base = os.path.join(data_dir, "cifar-10-batches-py")
    train = ndl.data.CIFAR10Dataset(base, train=True)
    test = ndl.data.CIFAR10Dataset(base, train=False)
    if train_subset:
        train.X, train.y = train.X[:train_subset], train.y[:train_subset]
    return train, test


DATASETS = {"mnist": (load_mnist, 784), "cifar10": (load_cifar10, 3072)}


def ptq_target(meta, device):
    """Rebuild a saved MLP and return (model, evaluate) for post-training quantization."""
    load, in_features = DATASETS[meta["dataset"]]
    _, test = load(meta["data_dir"], 0)
    model = make_mlp(QuantConfig(), meta["hidden"], device, in_features, meta.get("depth", 2))
    return model, lambda m: {"test_acc": evaluate(m, test, device)[1]}


def to_device(batch, device):
    """DataLoader builds tensors on the default device; copy them to `device`."""
    X, y = batch
    return ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)


def accuracy(logits, y):
    return float(np.mean(logits.numpy().argmax(axis=1) == y.numpy()))


def evaluate(model, dataset, device, batch_size=1000):
    """Average loss and accuracy over the whole dataset."""
    model.eval()
    loss_fn = nn.SoftmaxLoss()
    total_loss, total_correct, n = 0.0, 0.0, 0
    for batch in ndl.data.DataLoader(dataset, batch_size=batch_size, shuffle=False):
        X, y = to_device(batch, device)
        logits = model(X)
        b = y.shape[0]
        total_loss += float(loss_fn(logits, y).numpy()) * b
        total_correct += accuracy(logits, y) * b
        n += b
    model.train()
    return total_loss / n, total_correct / n


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def epoch_checkpoint(args, name):
    """Path of a run's end-of-epoch training state, or None when checkpoints are off.
    Named after the output file too, so jobs sharing a folder never read each other's state."""
    if not args.checkpoint_dir:
        return None
    job = os.path.splitext(os.path.basename(args.out))[0]
    return os.path.join(args.checkpoint_dir, f"{job}__{name}_s{args.seed}.npz")


def run_identity(args, name):
    """Everything that defines a run; a checkpoint only resumes the run it was saved by."""
    return json.dumps({"config": name, "seed": args.seed, "out": os.path.abspath(args.out),
                       "dataset": args.dataset, "train_subset": args.train_subset, "lr": args.lr,
                       "hidden": args.hidden, "depth": args.depth, "batch_size": args.batch_size,
                       "epochs": args.epochs}, sort_keys=True)


def checkpointed_epochs(args, name):
    """Epochs saved in this run's own checkpoint (0 if there is none)."""
    path = epoch_checkpoint(args, name)
    if not (path and os.path.exists(path)):
        return 0
    data = np.load(path)
    return int(data["epoch"]) if str(data["identity"]) == run_identity(args, name) else 0


def save_state(path, identity, epoch, step, model, optimizer, elapsed):
    """Everything needed to continue exactly: weights, SGD momentum, SR random state, counters.
    (The data order needs nothing: it is re-seeded at the start of every epoch.)"""
    params = model.parameters()
    arrays = {f"p{i}": p.numpy() for i, p in enumerate(params)}
    arrays.update({f"u{i}": optimizer.u[p].numpy() for i, p in enumerate(params) if p in optimizer.u})
    tmp = path + ".tmp.npz"
    np.savez(tmp, identity=identity, epoch=epoch, step=step, elapsed=elapsed, sr=json.dumps(sr_state()), **arrays)
    os.replace(tmp, path)   # atomic: an interruption never leaves a half-written checkpoint


def load_state(path, identity, model, optimizer, device):
    data = np.load(path)
    if str(data["identity"]) != identity:
        raise RuntimeError(f"{path} was saved by a different run; refusing to resume from it")
    for i, p in enumerate(model.parameters()):
        p.data = ndl.Tensor(data[f"p{i}"], device=device)
        if f"u{i}" in data:
            optimizer.u[p] = ndl.Tensor(data[f"u{i}"], device=device)
    set_sr_state(json.loads(str(data["sr"])))
    return int(data["epoch"]), int(data["step"]), float(data["elapsed"])


def train_one(config, args, train_set, test_set, device, writer, out_file=None):
    """Train one config (from scratch, or from its last end-of-epoch checkpoint) and stream
    its metrics to the CSV writer."""
    # Same seed for every config: identical initial weights and data order, so
    # differences between runs come only from FP4.
    np.random.seed(args.seed)
    seed_sr(args.seed)   # SR has its own random stream, restarted per run
    model = make_mlp(config, args.hidden, device, DATASETS[args.dataset][1], args.depth)
    name = config.name
    optimizer = ndl.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.SoftmaxLoss()
    start = time.time()
    first_epoch, step = 1, 0
    ckpt = epoch_checkpoint(args, name)
    if ckpt and os.path.exists(ckpt):
        done, step, elapsed = load_state(ckpt, run_identity(args, name), model, optimizer, device)
        first_epoch, start = done + 1, start - elapsed
        print(f"  {name}: continuing from the checkpoint after epoch {done}", flush=True)

    def log(split, epoch, step, loss, acc):
        writer.writerow({"config": name, "seed": args.seed, "split": split,
                         "epoch": epoch, "step": step, "loss": f"{loss:.6f}",
                         "accuracy": f"{acc:.4f}", "elapsed_s": f"{time.time() - start:.1f}"})

    for epoch in range(first_epoch, args.epochs + 1):
        # The DataLoader shuffles with np.random, which SR also draws from during
        # backward. Reseeding here gives every config the same batches each epoch.
        np.random.seed(args.seed * 1000 + epoch)
        loader = ndl.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        for batch in loader:
            X, y = to_device(batch, device)
            optimizer.reset_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_value = float(loss.numpy())
            log("train", epoch, step, loss_value, accuracy(logits, y))
            if not np.isfinite(loss_value):
                print(f"  {config.name}: loss became {loss_value} at step {step}; stopping run")
                return
        test_loss, test_acc = evaluate(model, test_set, device)
        log("test", epoch, step, test_loss, test_acc)
        print(f"  {name:24s} epoch {epoch}: test loss {test_loss:.4f}  "
              f"test acc {test_acc:.4f}  ({time.time() - start:.0f}s)", flush=True)
        if ckpt:
            out_file.flush()                 # this epoch's rows reach the disk before its checkpoint,
            os.fsync(out_file.fileno())      # so the CSV is never behind the checkpoint
            save_state(ckpt, run_identity(args, name), epoch, step, model, optimizer, time.time() - start)
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)   # the run is complete; its rows are in the CSV
    if args.save_weights:
        path = args.save_weights.format(config=name, seed=args.seed)
        save_checkpoint(model, path, {"runner": "mlp", "dataset": args.dataset, "hidden": args.hidden,
                                      "depth": args.depth, "data_dir": args.data_dir, "config": name})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", choices=CONFIG_SETS, default="grid")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="train only these configs from the chosen set, e.g. --only wgrad+sr")
    parser.add_argument("--device", choices=["cpu", "cuda", "cpu_numpy"], default="cpu")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="results/mnist.csv")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=0,
                        help="train on only the first N images (0 = all)")
    parser.add_argument("--dataset", choices=DATASETS, default="mnist")
    parser.add_argument("--resume", action="store_true",
                        help="keep configs already finished in --out and train only the rest")
    parser.add_argument("--checkpoint-dir", default="",
                        help="save training state after every epoch here, so --resume continues mid-run")
    parser.add_argument("--save-weights", default="",
                        help="checkpoint path, may use {config} and {seed}")
    args = parser.parse_args()

    device = getattr(ndl, args.device)()
    if not device.enabled():
        parser.error(f"needle's {args.device} backend is not built. On Colab: Runtime > "
                     "Change runtime type > GPU, then rebuild with `make`.")
    train_set, test_set = DATASETS[args.dataset][0](args.data_dir, args.train_subset)
    configs = CONFIG_SETS[args.configs]
    if args.only:
        unknown = set(args.only) - {c.name for c in configs}
        if unknown:
            parser.error(f"unknown config(s) {sorted(unknown)}; choose from {[c.name for c in configs]}")
        configs = [c for c in configs if c.name in args.only]
    print(f"Training {len(configs)} configs on {device}: {[c.name for c in configs]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    kept = []
    if args.resume and os.path.exists(args.out):
        # A config is finished when it has a test row for every epoch; drop partial runs.
        with open(args.out) as f:
            old = list(csv.DictReader(f))
        done = {c.name for c in configs
                if sum(r["config"] == c.name and r["split"] == "test" for r in old) == args.epochs}
        # Unfinished runs keep their rows up to their last checkpoint (none without checkpoints).
        upto = {c.name: (args.epochs if c.name in done else checkpointed_epochs(args, c.name)) for c in configs}
        kept = [r for r in old if r["config"] in upto and int(r["epoch"]) <= upto[r["config"]]]
        configs = [c for c in configs if c.name not in done]
        print(f"Resuming: {sorted(done)} already finished; training {[c.name for c in configs]}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(kept)
        for config in configs:
            train_one(config, args, train_set, test_set, device, writer, f)
            f.flush()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
