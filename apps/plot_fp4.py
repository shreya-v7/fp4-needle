# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
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
    color = FAMILY_COLORS.get(family, "#7f7f7f")   # gray for configs we have no color for
    return dict(color=color, linestyle="--" if config.endswith("+sr") else "-", linewidth=2)


def load(paths):
    """rows[config][seed] -> {"train": [(step, loss)], "test": [(epoch, loss, acc)], "time": s}

    If the same (config, seed) appears in more than one file (e.g. a partial run
    and its rerun), only the file with the most finished epochs is kept."""
    per_file = defaultdict(lambda: {"train": [], "test": [], "time": 0.0})  # (config, seed, path)
    for path in dict.fromkeys(paths):   # ignore a file passed twice
        with open(path) as f:
            for r in csv.DictReader(f):
                run = per_file[(r["config"], r["seed"], path)]
                if r["split"] == "train":
                    run["train"].append((int(r["step"]), float(r["loss"])))
                else:
                    run["test"].append((int(r["epoch"]), float(r["loss"]), float(r["accuracy"])))
                run["time"] = float(r["elapsed_s"])
    rows = defaultdict(dict)
    for (config, seed, path), run in per_file.items():
        kept = rows[config].get(seed)
        if kept is not None:
            print(f"warning: {config} seed {seed} is in several files; keeping the most complete run")
        if kept is None or len(run["test"]) > len(kept["test"]):
            rows[config][seed] = run
    return rows


def smooth(values, alpha=0.02):
    """Exponential moving average, so per-batch noise doesn't hide the trend."""
    out, avg = [], values[0]
    for v in values:
        avg = alpha * v + (1 - alpha) * avg
        out.append(avg)
    return np.array(out)


def plot(rows, configs, title, path, perplexity=False):
    """Training loss, loss relative to FP32, and per-epoch test accuracy (or, with
    perplexity=True for language models, test perplexity = exp(test loss))."""
    configs = [c for c in configs if c in rows]
    if not configs:
        print(f"no runs to plot for {path}; skipped")
        return
    fig, (ax_loss, ax_gap, ax_acc) = plt.subplots(1, 3, figsize=(18, 4.8))

    def mean_curve(c):
        """Smooth each seed's train loss, then average seeds (cut to the shortest run).
        Seeds that stopped early (diverged, no test row) are left out if others finished."""
        seeds = [s for s in rows[c].values() if s["test"]] or list(rows[c].values())
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

        # Test metric per epoch: mean over seeds, bars show min..max.
        epochs = sorted({e for s in seeds for e, _, _ in s["test"]})
        if perplexity:
            vals = [[np.exp(l) for s in seeds for e2, l, _ in s["test"] if e2 == e] for e in epochs]
        else:
            vals = [[a * 100 for s in seeds for e2, _, a in s["test"] if e2 == e] for e in epochs]
        mean = np.array([np.mean(v) for v in vals])
        lo = mean - np.array([np.min(v) for v in vals])
        hi = np.array([np.max(v) for v in vals]) - mean
        ax_acc.errorbar(epochs, mean, yerr=[lo, hi], marker="o", markersize=8, capsize=3,
                        label=c, **style(c))

    if perplexity:   # language-model losses sit in a narrow range (about 4 to 9)
        ax_loss.set(xlabel="training step", ylabel="train loss (smoothed)", title="Training loss")
    else:
        ax_loss.set(xlabel="training step", ylabel="train loss (smoothed)", yscale="log",
                    title="Training loss")
        ax_loss.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_loss.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax_loss.set_yticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    ax_gap.set(xlabel="training step", ylabel="train loss vs FP32 (%)",
               title="Training loss relative to FP32")
    if perplexity:
        ax_acc.set(xlabel="epoch", ylabel="test perplexity (lower is better)",
                   title="Test perplexity (mean, min-max over seeds)")
    else:
        ax_acc.set(xlabel="epoch", ylabel="test accuracy (%)",
                   title="Test accuracy (mean, min-max over seeds)")
    ax_acc.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in (ax_loss, ax_gap, ax_acc):
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ax_acc.legend(fontsize=8, frameon=False, loc="upper right" if perplexity else "lower right")
    num_seeds = max(len(rows[c]) for c in configs)
    fig.suptitle(f"{title}  ({num_seeds} seed{'s' if num_seeds > 1 else ''})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def summary(rows, path, perplexity=False):
    ppl_head, ppl_sep = (" final test ppl |", "---|") if perplexity else ("", "")
    lines = [f"| config | seeds | final test acc (%) | final test loss |{ppl_head} "
             "train loss, last 100 steps | s per run |",
             f"|---|---|---|---|{ppl_sep}---|---|"]
    for c in [c for c in GRID + ABLATION[1:] if c in rows]:
        seeds = list(rows[c].values())
        acc = [s["test"][-1][2] * 100 for s in seeds if s["test"]]
        tl = [s["test"][-1][1] for s in seeds if s["test"]]
        tr = [np.mean([l for _, l in s["train"][-100:]]) for s in seeds]
        t = [s["time"] for s in seeds]
        if len(acc) < len(seeds):  # a run stopped early because its loss became inf/nan
            lines.append(f"| {c} | {len(seeds)} | diverged in {len(seeds) - len(acc)} seed(s) | | | |")
            continue
        ppl = np.exp(tl)
        ppl_cell = f" {np.mean(ppl):.1f} ± {np.std(ppl):.1f} |" if perplexity else ""
        lines.append(f"| {c} | {len(seeds)} | {np.mean(acc):.2f} ± {np.std(acc):.2f} | "
                     f"{np.mean(tl):.4f} |{ppl_cell} {np.mean(tr):.4f} | {np.mean(t):.0f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", path)


def plot_grad_error(csv_path, path):
    """Bar chart of apps/fp4_grad_error.py output: one panel per training checkpoint,
    one bar per config (relative weight-gradient error vs FP32, mean over layers).
    SR bars are hatched; the dot on them is the error of the mean of K SR draws,
    i.e. what remains after averaging away SR's noise (its bias)."""
    data = defaultdict(lambda: defaultdict(list))   # data[step][config] -> rows
    for r in csv.DictReader(open(csv_path)):
        data[int(r["step"])][r["config"]].append(r)
    steps = sorted(data)
    fig, axes = plt.subplots(1, len(steps), figsize=(5.5 * len(steps), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, step in zip(axes, steps):
        configs = list(data[step])
        for y, c in enumerate(configs):
            err = np.mean([float(r["rel_error"]) for r in data[step][c]]) * 100
            err_mean = np.mean([float(r["rel_error_of_mean"]) for r in data[step][c]]) * 100
            sr = c.endswith("+sr")
            ax.barh(y, err, height=0.7, color=style(c)["color"], hatch="//" if sr else None,
                    edgecolor="white", linewidth=0)
            ax.annotate(f"{err:.0f}%", (err, y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#333333")
            if sr:
                ax.plot(err_mean, y, "o", color="#222222", markersize=8)
        ax.set_yticks(range(len(configs)), configs)
        ax.invert_yaxis()
        ax.set(xlabel="weight-gradient error vs FP32 (%)", title=f"after {step} FP32 training steps")
        ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    sr_draws = max(int(r["sr_draws"]) for d in data.values() for rs in d.values() for r in rs)
    fig.suptitle(f"Gradient error from each FP4 matmul (bars: one draw; dots: mean of {sr_draws} SR draws)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--grad-error", help="CSV from apps/fp4_grad_error.py to plot as well")
    parser.add_argument("--title", default="MNIST MLP", help="model name used in plot titles")
    parser.add_argument("--prefix", default="mnist", help="file name prefix of the two plots")
    parser.add_argument("--perplexity", action="store_true",
                        help="language model: plot and tabulate test perplexity")
    args = parser.parse_args()
    rows = load(args.csvs)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.grad_error:
        plot_grad_error(args.grad_error, os.path.join(args.out_dir, "grad_error.png"))
    plot(rows, GRID, f"{args.title}: FP4 matmuls added one at a time",
         os.path.join(args.out_dir, f"{args.prefix}_grid.png"), args.perplexity)
    plot(rows, ABLATION, f"{args.title}: one backward matmul in FP4",
         os.path.join(args.out_dir, f"{args.prefix}_ablation.png"), args.perplexity)
    summary(rows, os.path.join(args.out_dir, "summary.md"), args.perplexity)


if __name__ == "__main__":
    main()
