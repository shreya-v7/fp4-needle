# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Verdicts for the extended experiments, applying docs/PREREGISTRATION.md exactly.

Reads the CSVs written by fp4_experiment.py from one folder, named
    main_seed<S>_*.csv       main run (20 epochs, configs split over several files)
    lr<LR>_seed<S>.csv       learning-rate sweep (10 epochs)
    depth<D>_seed<S>.csv     depth sweep (10 epochs)
and writes a markdown report with the pre-registered verdicts.

    python3 apps/fp4_stats.py docs/results/extended --out docs/results/extended/verdicts.md
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

CHANCE = 10.0   # accuracy (%) counted for epochs after a run's loss became non-finite


def load(paths):
    """test accuracy (%) per epoch for every (config, seed): {(config, seed): [acc_1, acc_2, ...]}"""
    runs = defaultdict(dict)
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["split"] == "test":
                    runs[(r["config"], int(r["seed"]))][int(r["epoch"])] = 100 * float(r["accuracy"])
    return {k: [v[e] for e in sorted(v)] for k, v in runs.items()}


def metric_a(accs, epochs):
    """Mean test accuracy over the last 3 of `epochs` epochs; missing epochs count as chance."""
    accs = (accs + [CHANCE] * epochs)[:epochs]
    return float(np.mean(accs[-3:]))


def by_seed(runs, config, epochs):
    return {s: metric_a(a, epochs) for (c, s), a in runs.items() if c == config}


def paired(a, b):
    """Paired differences a - b over the seeds both have."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return {"n": len(d), "mean": float(d.mean()), "sd": sd, "se": sd / np.sqrt(len(d)),
            "pos": int((d > 0).sum()), "d": d}


def fmt(p):
    return (f"mean {p['mean']:+.2f} points, sd {p['sd']:.2f}, paired SE {p['se']:.2f}, "
            f"positive in {p['pos']}/{p['n']} seeds")


def main_report(runs, epochs=20):
    configs = sorted({c for c, _ in runs}, key=lambda c: (c != "fp32", c))
    A = {c: by_seed(runs, c, epochs) for c in configs}
    lines = [f"## Main run ({epochs} epochs, metric A = mean test accuracy of the last 3 epochs)", "",
             "| config | seeds | metric A (%) | final epoch (%) |", "|---|---|---|---|"]
    for c in configs:
        a = np.array(list(A[c].values()))
        fin = np.array([(runs[(c, s)] + [CHANCE] * epochs)[epochs - 1] for s in A[c]])
        lines.append(f"| {c} | {len(a)} | {a.mean():.2f} ± {a.std(ddof=1) if len(a) > 1 else 0:.2f} | "
                     f"{fin.mean():.2f} |")

    def need(*cs):
        return all(c in A for c in cs)

    lines += ["", "### Pre-registered verdicts", ""]
    if need("dgrad", "dgrad+sr", "fprop+dgrad+wgrad", "fprop+dgrad+wgrad+sr"):
        ok = True
        for rtn, sr in [("dgrad", "dgrad+sr"), ("fprop+dgrad+wgrad", "fprop+dgrad+wgrad+sr")]:
            p = paired(A[sr], A[rtn])
            ok &= p["mean"] > 2 * p["se"] and p["pos"] >= 0.8 * p["n"]
            lines.append(f"* {sr} vs {rtn}: {fmt(p)}")
        lines.append(f"* **H1 (round-to-nearest gradients hurt, SR fixes them): {'SUPPORTED' if ok else 'NOT SUPPORTED'}**")
    if need("fp32", "dgrad+sr", "fprop+dgrad+wgrad+sr"):
        ok = True
        for sr in ("dgrad+sr", "fprop+dgrad+wgrad+sr"):
            p = paired(A[sr], A["fp32"])
            ok &= abs(p["mean"]) < 1
            lines.append(f"* {sr} vs fp32: {fmt(p)}")
        lines.append(f"* **H1b (SR recovers FP32, within 1 point): {'SUPPORTED' if ok else 'NOT SUPPORTED'}**")
    if need("fp32", "dgrad-dy", "dgrad-w", "dgrad-dy+sr"):
        pa = paired(A["dgrad-dy"], A["fp32"])
        pb = paired(A["dgrad-w"], A["fp32"])
        pc = paired(A["dgrad-dy+sr"], A["fp32"])
        a_ok = pa["mean"] < -1 and -pa["mean"] > 2 * pa["se"]
        lines += [f"* (a) dgrad-dy vs fp32: {fmt(pa)} -> {'yes' if a_ok else 'no'}",
                  f"* (b) dgrad-w vs fp32: {fmt(pb)} -> {'yes' if abs(pb['mean']) < 1 else 'no'}",
                  f"* (c) dgrad-dy+sr vs fp32: {fmt(pc)} -> {'yes' if abs(pc['mean']) < 1 else 'no'}"]
        ok = a_ok and abs(pb["mean"]) < 1 and abs(pc["mean"]) < 1
        lines.append(f"* **H2 (it is the rounding of dY, not of W): {'SUPPORTED' if ok else 'NOT SUPPORTED'}**")
    if need("fp32", "dgrad"):
        gap = lambda e: np.mean([(runs[("fp32", s)] + [CHANCE] * epochs)[e - 1]
                                 - (runs[("dgrad", s)] + [CHANCE] * epochs)[e - 1] for s in A["dgrad"]])
        g3, g20 = gap(3), gap(epochs)
        verdict = ("genuine failure" if g20 >= g3 else "slower learning" if g20 < g3 / 2 else "partly closes")
        lines.append(f"* **H3: FP32 minus dgrad is {g3:.2f} points at epoch 3 and {g20:.2f} at epoch {epochs}: "
                     f"{verdict}**")
        lines += ["", "Gap to FP32 by epoch (mean over seeds, points):", "",
                  "| epoch | " + " | ".join(c for c in configs if c != "fp32") + " |",
                  "|---|" + "---|" * (len(configs) - 1)]
        for e in range(1, epochs + 1):
            cells = []
            for c in configs:
                if c == "fp32":
                    continue
                g = [(runs[(c, s)] + [CHANCE] * epochs)[e - 1] - (runs[("fp32", s)] + [CHANCE] * epochs)[e - 1]
                     for s in A[c] if s in A["fp32"]]
                cells.append(f"{np.mean(g):+.1f}")
            lines.append(f"| {e} | " + " | ".join(cells) + " |")
    return lines


def sweep_report(title, groups, main_runs, main_value, epochs=10):
    """groups: {setting value: runs}; the main run supplies the main_value setting (its first epochs)."""
    lines = ["", f"## {title} ({epochs} epochs, metric A over epochs {epochs - 2} to {epochs})", "",
             "| setting | fp32 (%) | dgrad (%) | dgrad+sr (%) | dgrad+sr minus dgrad | holds? |",
             "|---|---|---|---|---|---|"]
    trimmed = {k: v[:epochs] for k, v in main_runs.items() if k[0] in ("fp32", "dgrad", "dgrad+sr")}
    for value, runs in sorted({**groups, main_value: trimmed}.items(), key=lambda kv: float(kv[0])):
        A = {c: by_seed(runs, c, epochs) for c in ("fp32", "dgrad", "dgrad+sr")}
        if not all(A.values()):
            continue
        p = paired(A["dgrad+sr"], A["dgrad"])
        holds = p["mean"] > 0 and p["mean"] > 2 * p["se"]
        mean = lambda c: np.mean(list(A[c].values()))
        lines.append(f"| {value} | {mean('fp32'):.2f} | {mean('dgrad'):.2f} | {mean('dgrad+sr'):.2f} | "
                     f"{p['mean']:+.2f} (SE {p['se']:.2f}, {p['pos']}/{p['n']} seeds) | {'yes' if holds else 'no'} |")
    return lines


def plot_curves(runs, path, epochs=20):
    """Test accuracy by epoch, mean over seeds with a min-max band, one panel per question."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Round to nearest vs SR", ["fp32", "dgrad", "dgrad+sr", "fprop+dgrad+wgrad", "fprop+dgrad+wgrad+sr"]),
              ("Which Dgrad operand: dY or W?", ["fp32", "dgrad-dy", "dgrad-dy+sr", "dgrad-w"])]
    colors = {"fp32": "#2a78d6", "dgrad": "#e87ba4", "dgrad-dy": "#eb6834", "dgrad-w": "#6250d6",
              "fprop+dgrad+wgrad": "#eda100"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, (title, configs) in zip(axes, panels):
        for c in configs:
            seeds = [(runs[k] + [CHANCE] * epochs)[:epochs] for k in runs if k[0] == c]
            if not seeds:
                continue
            a = np.array(seeds)
            e = np.arange(1, epochs + 1)
            base = c[:-3] if c.endswith("+sr") else c
            style = dict(color=colors.get(base, "#7f7f7f"), linestyle="--" if c.endswith("+sr") else "-")
            ax.plot(e, a.mean(0), label=c, linewidth=2, **style)
            ax.fill_between(e, a.min(0), a.max(0), alpha=0.12, color=style["color"])
        ax.set(title=title, xlabel="epoch", ylabel="test accuracy (%)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"CIFAR-10 MLP, {epochs} epochs: mean over seeds, band = min to max")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = glob.glob(os.path.join(args.folder, "*.csv"))
    main_runs = load([f for f in files if os.path.basename(f).startswith("main_")])
    lr = defaultdict(list)
    depth = defaultdict(list)
    for f in files:
        m = re.match(r"lr([0-9.]+)_seed", os.path.basename(f))
        n = re.match(r"depth(\d+)_seed", os.path.basename(f))
        if m:
            lr[m.group(1)].append(f)
        if n:
            depth[n.group(1)].append(f)
    lines = ["# Extended experiments: verdicts", "",
             "Criteria from `docs/PREREGISTRATION.md`, applied by `apps/fp4_stats.py`.", ""]
    if main_runs:
        plot_curves(main_runs, os.path.splitext(args.out)[0] + ".png")
        lines += main_report(main_runs)
        lines += sweep_report("Learning-rate sweep", {k: load(v) for k, v in lr.items()}, main_runs, "0.05")
        lines += sweep_report("Depth sweep (hidden layers)", {k: load(v) for k, v in depth.items()}, main_runs, "2")
    text = "\n".join(lines) + "\n"
    open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
