"""Stage 3 experiment runner: train an MLP on MNIST with FP4 linear layers.

Each run trains the same MLP (same seed, same data order) with a different
QuantConfig and logs to one CSV file:
    * one row per training step:   split=train, the batch loss and accuracy
    * one row per epoch:           split=test,  loss and accuracy on the test set

Examples (from the repo root):
    # the 6 distinct configs of the project plan, short CPU run
    python3 apps/fp4_experiment.py --configs grid --device cpu --epochs 1 --train-subset 20000
    # every config incl. single-matmul ablations, full data, on a GPU
    python3 apps/fp4_experiment.py --configs all --device cuda --epochs 3
    # then plot
    python3 apps/plot_fp4.py results/mnist.csv

NumPy is used here only for data loading and metrics (argmax / accuracy),
like the homework's apps/simple_ml.py; all model math runs in needle.
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

CONFIG_SETS = {"grid": GRID, "ablation": ABLATION, "all": GRID + ABLATION}

CSV_FIELDS = ["config", "seed", "split", "epoch", "step", "loss", "accuracy", "elapsed_s"]


# ----------------------------------------------------------------------------
# Model and data
# ----------------------------------------------------------------------------

def make_mlp(config, hidden, device):
    """784 -> hidden -> hidden -> 10 MLP where every linear layer uses `config`."""
    return nn.Sequential(
        nn.Flatten(),
        nn.QuantizedLinear(784, hidden, device=device, config=config), nn.ReLU(),
        nn.QuantizedLinear(hidden, hidden, device=device, config=config), nn.ReLU(),
        nn.QuantizedLinear(hidden, 10, device=device, config=config),
    )


def load_mnist(data_dir, train_subset):
    train = ndl.data.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                                  os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test = ndl.data.MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                                 os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    if train_subset:  # use the first N training images for quick runs
        train.images, train.labels = train.images[:train_subset], train.labels[:train_subset]
    return train, test


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

def train_one(config, args, train_set, test_set, device, writer):
    """Train one config from scratch and stream its metrics to the CSV writer."""
    # Same seed for every config: identical initial weights and data order, so
    # differences between runs come only from FP4.
    np.random.seed(args.seed)
    model = make_mlp(config, args.hidden, device)
    optimizer = (ndl.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam"
                 else ndl.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9))
    loss_fn = nn.SoftmaxLoss()
    start = time.time()

    def log(split, epoch, step, loss, acc):
        writer.writerow({"config": config.name, "seed": args.seed, "split": split,
                         "epoch": epoch, "step": step, "loss": f"{loss:.6f}",
                         "accuracy": f"{acc:.4f}", "elapsed_s": f"{time.time() - start:.1f}"})

    step = 0
    for epoch in range(1, args.epochs + 1):
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
        print(f"  {config.name:24s} epoch {epoch}: test loss {test_loss:.4f}  "
              f"test acc {test_acc:.4f}  ({time.time() - start:.0f}s)", flush=True)


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
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=0,
                        help="train on only the first N images (0 = all 60000)")
    args = parser.parse_args()

    device = getattr(ndl, args.device)()
    if not device.enabled():
        parser.error(f"needle's {args.device} backend is not built. On Colab: Runtime > "
                     "Change runtime type > GPU, then rebuild with `make`.")
    train_set, test_set = load_mnist(args.data_dir, args.train_subset)
    configs = CONFIG_SETS[args.configs]
    if args.only:
        unknown = set(args.only) - {c.name for c in configs}
        if unknown:
            parser.error(f"unknown config(s) {sorted(unknown)}; choose from {[c.name for c in configs]}")
        configs = [c for c in configs if c.name in args.only]
    print(f"Training {len(configs)} configs on {device}: {[c.name for c in configs]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for config in configs:
            train_one(config, args, train_set, test_set, device, writer)
            f.flush()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
