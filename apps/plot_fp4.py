"""Plot Stage 3 results written by apps/fp4_experiment.py.

    python3 apps/plot_fp4.py results/mnist_seed*.csv --out-dir results

Writes, averaging over every seed found in the CSVs:
    mnist_grid.png       train loss and test accuracy for the project grid
    mnist_ablation.png   the same for single-matmul ablations (vs FP32)
    summary.md           a table of final numbers per config (the chart's table view)
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # no display needed (Colab, CI, servers)
import matplotlib.pyplot as plt
import numpy as np

# Color follows the config family; SR runs reuse their family's color with a
# dashed line, so SR on/off is readable without color.
FAMILY_COLORS = {
    "fp32": "#2a78d6",                # blue
    "fprop": "#eb6834",               # orange
    "fprop+dgrad": "#1baf7a",         # aqua
    "fprop+dgrad+wgrad": "#eda100",   # yellow
    "dgrad": "#e87ba4",               # magenta
    "wgrad": "#6250d6",               # violet
}
GRID = ["fp32", "fprop", "fprop+dgrad", "fprop+dgrad+sr", "fprop+dgrad+wgrad", "fprop+dgrad+wgrad+sr"]
ABLATION = ["fp32", "dgrad", "dgrad+sr", "wgrad", "wgrad+sr"]


def style(config):
    family = config[:-3] if config.endswith("+sr") else config
    return dict(color=FAMILY_COLORS[family], linestyle="--" if config.endswith("+sr") else "-",
                linewidth=2)


def load(paths):
    """rows[config][seed] -> {"train": [(step, loss)], "test": [(epoch, loss, acc)], "time": s}"""
    rows = defaultdict(lambda: defaultdict(lambda: {"train": [], "test": [], "time": 0.0}))
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                run = rows[r["config"]][r["seed"]]
                if r["split"] == "train":
                    run["train"].append((int(r["step"]), float(r["loss"])))
                else:
                    run["test"].append((int(r["epoch"]), float(r["loss"]), float(r["accuracy"])))
                run["time"] = float(r["elapsed_s"])
    return rows


def smooth(values, alpha=0.02):
    """Exponential moving average, so per-batch noise doesn't hide the trend."""
    out, avg = [], values[0]
    for v in values:
        avg = alpha * v + (1 - alpha) * avg
        out.append(avg)
    return np.array(out)


def plot(rows, configs, title, path):
    configs = [c for c in configs if c in rows]
    fig, (ax_loss, ax_gap, ax_acc) = plt.subplots(1, 3, figsize=(18, 4.8))

    def mean_curve(c):
        """Smooth each seed's train loss, then average seeds (cut to the shortest run)."""
        seeds = list(rows[c].values())
        n = min(len(s["train"]) for s in seeds)
        return np.mean([smooth([l for _, l in s["train"][:n]]) for s in seeds], axis=0)

    baseline = mean_curve("fp32") if "fp32" in rows else None
    for c in configs:
        seeds = list(rows[c].values())
        curve = mean_curve(c)
        steps = np.arange(1, len(curve) + 1)
        ax_loss.plot(steps, curve, label=c, **style(c))

        # Gap to FP32 in percent: small differences that overlap in the loss
        # panel become visible here. FP32 itself is the 0% line.
        if baseline is not None:
            n = min(len(curve), len(baseline))
            gap = (curve[:n] / baseline[:n] - 1) * 100
            ax_gap.plot(steps[:n], gap, label=c, **style(c))
            ax_gap.annotate(c, (n, gap[-1]), xytext=(4, 0), textcoords="offset points",
                            fontsize=8, va="center", color="#333333")

        # Test accuracy per epoch: mean over seeds, bars show min..max.
        epochs = sorted({e for s in seeds for e, _, _ in s["test"]})
        accs = [[a for s in seeds for e2, _, a in s["test"] if e2 == e] for e in epochs]
        mean = np.array([np.mean(a) for a in accs]) * 100
        lo = mean - np.array([np.min(a) for a in accs]) * 100
        hi = np.array([np.max(a) for a in accs]) * 100 - mean
        ax_acc.errorbar(epochs, mean, yerr=[lo, hi], marker="o", markersize=8, capsize=3,
                        label=c, **style(c))

    ax_loss.set(xlabel="training step", ylabel="train loss (smoothed)", yscale="log",
                title="Training loss")
    ax_loss.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_loss.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_loss.set_yticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    ax_gap.set(xlabel="training step", ylabel="train loss vs FP32 (%)",
               title="Training loss relative to FP32")
    ax_acc.set(xlabel="epoch", ylabel="test accuracy (%)", title="Test accuracy (mean, min-max over seeds)")
    ax_acc.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in (ax_loss, ax_gap, ax_acc):
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ax_acc.legend(fontsize=8, frameon=False, loc="lower right")
    num_seeds = max(len(rows[c]) for c in configs)
    fig.suptitle(f"{title}  ({num_seeds} seed{'s' if num_seeds > 1 else ''})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def summary(rows, path):
    lines = ["| config | seeds | final test acc (%) | final test loss | train loss, last 100 steps | s per run |",
             "|---|---|---|---|---|---|"]
    for c in [c for c in GRID + ABLATION[1:] if c in rows]:
        seeds = list(rows[c].values())
        acc = [s["test"][-1][2] * 100 for s in seeds if s["test"]]
        tl = [s["test"][-1][1] for s in seeds if s["test"]]
        tr = [np.mean([l for _, l in s["train"][-100:]]) for s in seeds]
        t = [s["time"] for s in seeds]
        lines.append(f"| {c} | {len(seeds)} | {np.mean(acc):.2f} ± {np.std(acc):.2f} | "
                     f"{np.mean(tl):.4f} | {np.mean(tr):.4f} | {np.mean(t):.0f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    rows = load(args.csvs)
    os.makedirs(args.out_dir, exist_ok=True)
    plot(rows, GRID, "MNIST MLP: FP4 matmuls added one at a time",
         os.path.join(args.out_dir, "mnist_grid.png"))
    plot(rows, ABLATION, "MNIST MLP: one backward matmul in FP4",
         os.path.join(args.out_dir, "mnist_ablation.png"))
    summary(rows, os.path.join(args.out_dir, "summary.md"))


if __name__ == "__main__":
    main()
