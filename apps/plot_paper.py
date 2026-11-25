# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Figures and tables for the paper experiments (E2 threshold, E3 data size, E4 model size, PTQ).

    python3 apps/plot_paper.py threshold results/paper/gpt_threshold*.csv --out results/paper/fig_threshold.png
    python3 apps/plot_paper.py datasize  results/paper/lstm_L*_seed*.csv  --out results/paper/fig_datasize.png
    python3 apps/plot_paper.py scaling   results/paper/gpt_*_seed*.csv    --out results/paper/fig_scaling.png
    python3 apps/plot_paper.py ptq       results/paper/storage/ptq.csv    --out results/paper/ptq.md
"""
import argparse
import csv
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"fp32": "#2a78d6", "fprop": "#eb6834", "fprop+dgrad": "#1baf7a",
          "fprop+dgrad+wgrad": "#eda100", "fprop+dgrad+wgrad+sr": "#e87ba4"}
SQRT3 = math.sqrt(3)


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def color(name):
    return COLORS.get(name.split("+switch")[0], "#6250d6")


def style(ax, title, xlabel, ylabel):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)


def final_test(rows, metric):
    """Last test row per (config, seed) -> value; metric 'ppl', 'bpc' or 'acc'."""
    out = {}
    for r in rows:
        if r["split"] == "test":
            loss = float(r["loss"])
            v = {"ppl": math.exp(loss), "bpc": loss / math.log(2), "acc": float(r["accuracy"])}[metric]
            out[(r["config"], r["seed"])] = v
    return out


def threshold(paths, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    train = defaultdict(list)
    for p in paths:
        if p.endswith("_gnr.csv"):
            by = defaultdict(list)
            for r in read(p):
                if r["ratio"] != "inf":
                    by[r["config"]].append((int(r["step"]), float(r["ratio"])))
            for c, pts in by.items():
                pts.sort()
                a1.plot([s for s, _ in pts], [v for _, v in pts], marker="o", markersize=3, label=c,
                        color=color(c), linestyle="--" if "switch" in c else "-")
        else:
            for r in read(p):
                if r["split"] == "train":
                    train[r["config"]].append((int(r["step"]), float(r["loss"])))
    a1.axhline(SQRT3, color="#222222", linestyle=":", linewidth=1.2)
    a1.annotate("sqrt(3)", (a1.get_xlim()[1], SQRT3), ha="right", va="bottom", fontsize=8)
    a1.set_yscale("log")
    style(a1, "Gradient-to-noise ratio r = ||g|| / ||g_q - g||", "training step", "r (log scale)")
    a1.legend(frameon=False, fontsize=8)
    if "fp32" in train:
        base = dict(train["fp32"])
        for c, pts in train.items():
            if c == "fp32":
                continue
            steps = [s for s, _ in pts if s in base]
            gap = np.array([dict(pts)[s] - base[s] for s in steps])
            k = 25
            smooth = np.convolve(gap, np.ones(k) / k, mode="valid") if len(gap) > k else gap
            a2.plot(steps[len(steps) - len(smooth):], smooth, label=c, color=color(c),
                    linestyle="--" if "switch" in c else "-")
        a2.axhline(0, color="#2a78d6", linewidth=1)
    style(a2, "Training loss minus FP32 (moving average)", "training step", "loss gap")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def sweep(paths, out, pattern, metric, xlabel, title, xlog):
    """Final metric per config vs a value parsed from each file name (data size or model size)."""
    vals = defaultdict(lambda: defaultdict(list))
    for p in paths:
        m = re.search(pattern, os.path.basename(p))
        if not m:
            continue
        x = m.group(1)
        for (c, _), v in final_test(read(p), metric).items():
            vals[c][x].append(v)
    xs = sorted({x for c in vals for x in vals[c]}, key=lambda s: float(s) if s.replace(".", "").isdigit() else s)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    lines = [f"| config | " + " | ".join(xs) + " |", "|---|" + "---|" * len(xs)]
    for c in vals:
        means = [np.mean(vals[c][x]) if vals[c][x] else np.nan for x in xs]
        a1.plot(range(len(xs)), means, marker="o", label=c, color=color(c))
        lines.append(f"| {c} | " + " | ".join(f"{m:.3f}" if not np.isnan(m) else "" for m in means) + " |")
        if "fp32" in vals and c != "fp32":
            base = [np.mean(vals["fp32"][x]) if vals["fp32"][x] else np.nan for x in xs]
            a2.plot(range(len(xs)), (np.array(means) / np.array(base) - 1) * 100, marker="o", label=c, color=color(c))
    for ax in (a1, a2):
        ax.set_xticks(range(len(xs)), xs)
    style(a1, title, xlabel, f"final test {metric}")
    style(a2, "Gap to FP32", xlabel, f"{metric} vs FP32 (%)")
    a2.axhline(0, color="#2a78d6", linewidth=1)
    a1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    open(os.path.splitext(out)[0] + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


def ptq(paths, out):
    rows = [r for p in paths for r in read(p)]
    lines = ["| checkpoint | metric | format | bits | no rotation | rotation 16 | rotation 128 |",
             "|---|---|---|---|---|---|---|"]
    table = defaultdict(dict)
    for r in rows:
        table[(r["checkpoint"], r["metric"], r["format"], r["bits"])][r["rotation"]] = float(r["value"])
    for (ck, m, fmt, bits), v in table.items():
        lines.append(f"| {ck} | {m} | {fmt} | {float(bits):.3f} | " +
                     " | ".join(f"{v[k]:.4f}" if k in v else "" for k in ("0", "16", "128")) + " |")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["threshold", "datasize", "scaling", "ptq"])
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.kind == "threshold":
        threshold(a.paths, a.out)
    elif a.kind == "datasize":
        sweep(a.paths, a.out, r"_L(\d+)_", "ppl", "PTB lines used for training (0 = all 42068)",
              "LSTM: test perplexity vs data size", False)
    elif a.kind == "scaling":
        sweep(a.paths, a.out, r"gpt_(tiny|small|medium)_", "bpc", "model size",
              "Mini-GPT: validation bits per character vs model size", False)
    else:
        ptq(a.paths, a.out)


if __name__ == "__main__":
    main()
