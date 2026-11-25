# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Storage study: error per bit of 4-bit weight formats, with and without rotation.

For every matmul weight in the given checkpoints (from the runners' --save-weights)
and, optionally, GPT-2's published weights, it records the crest factor and the
quantization error (NMSE) of each format in fp4/formats.py, with and without a
random Hadamard rotation. With --ptq it also loads each format's weights back into
the needle model and measures test accuracy or perplexity.

    python3 apps/fp4_storage.py --checkpoints results/ckpt/*.npz --gpt2 data/gpt2/model.safetensors \\
        --ptq --out-dir results/storage
"""
import argparse
import csv
import importlib
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.append(os.path.dirname(__file__))

import numpy as np

import needle as ndl
from needle.fp4 import formats as F
from fp4_common import assign_weights, is_matmul_weight, load_checkpoint, read_safetensors

RUNNERS = {"mlp": "fp4_experiment", "lstm": "fp4_lstm_experiment"}
GPT2_KEEP = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def quantize_weight(w, fmt, rot_size):
    """needle and GPT-2 store W as (in, out) and reduce over `in`, so block along axis 0."""
    wt = w.T
    q = F.quantize(wt, fmt) if not rot_size else F.quantize_rotated(wt, fmt, rot_size)
    return q.T


def tensor_rows(source, name, w, fmts, rot_sizes):
    wt = w.T
    rows = []
    kappa = {g: F.crest_factor_sq(wt, g) for g in sorted({f.block for f in fmts})}
    for rot in [0] + rot_sizes:
        wr = wt if not rot else F.Rotation(wt.shape[1], rot).forward(wt)
        kappa_r = {g: F.crest_factor_sq(wr, g) for g in kappa}
        for f in fmts:
            q = quantize_weight(w, f, rot)
            rows.append({"source": source, "tensor": name, "rows": w.shape[1], "cols": w.shape[0],
                         "format": f.name, "bits": f.bits, "rotation": rot,
                         "kappa_sq": f"{kappa_r[f.block]:.4f}", "nmse": f"{F.nmse(q, w):.6g}"})
    return rows


def ptq_rows(path, fmts, rot_sizes, device):
    meta, arrays = load_checkpoint(path)
    runner = importlib.import_module(RUNNERS[meta["runner"]])
    model, evaluate = runner.ptq_target(meta, device)
    out = []
    for rot in [0] + rot_sizes:
        for f in [None] + fmts:
            if f is None and rot:
                continue
            q = {k: (quantize_weight(v, f, rot) if f is not None and is_matmul_weight(k, v) else v)
                 for k, v in arrays.items()}
            assign_weights(model, q, device)
            metrics = evaluate(model)
            for metric, value in metrics.items():
                out.append({"checkpoint": os.path.basename(path), "trained_as": meta["config"],
                            "format": "fp32" if f is None else f.name,
                            "bits": 32 if f is None else f.bits, "rotation": rot,
                            "metric": metric, "value": f"{value:.5f}"})
            print(f"  {os.path.basename(path)} {('fp32' if f is None else f.name):10s} rot={rot:<4d} {metrics}",
                  flush=True)
    return out


def summarize(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(r["format"] for r in rows))
    rots = sorted({int(r["rotation"]) for r in rows})
    lines = ["| format | bits | " + " | ".join(f"SQNR dB, rot {r or 'none'}" for r in rots) + " |",
             "|---|---|" + "---|" * len(rots)]
    for n in names:
        cells = []
        for r in rots:
            e = [float(x["nmse"]) for x in rows if x["format"] == n and int(x["rotation"]) == r]
            cells.append(f"{-10 * math.log10(np.mean(e)):.2f}")
        bits = next(x["bits"] for x in rows if x["format"] == n)
        lines.append(f"| {n} | {float(bits):.3f} | " + " | ".join(cells) + " |")

    int4 = [x for x in rows if x["format"].startswith("int4") and int(x["rotation"]) == 0]
    pred = np.array([F.predicted_nmse_int4(float(x["kappa_sq"])) for x in int4])
    meas = np.array([float(x["nmse"]) for x in int4])
    if len(int4) > 1:
        corr = np.corrcoef(np.log(pred), np.log(meas))[0, 1]
        lines.append(f"\nINT4 error model kappa^2/588: median measured/predicted = "
                     f"{np.median(meas / pred):.2f}, log-log correlation = {corr:.3f} over {len(int4)} tensors")

    fp, it = ({(x["source"], x["tensor"]): x for x in rows if x["format"] == n and int(x["rotation"]) == 0}
              for n in ("nvfp4", "nvint4"))
    keys = [k for k in fp if k in it]
    if keys:
        kap = np.array([float(fp[k]["kappa_sq"]) for k in keys])
        win = np.array([float(it[k]["nmse"]) < float(fp[k]["nmse"]) for k in keys])
        lines.append(f"NVINT4 beats NVFP4 (no rotation) on {win.mean() * 100:.0f}% of {len(keys)} tensors; "
                     f"median kappa^2 where INT wins {np.median(kap[win]) if win.any() else float('nan'):.2f}, "
                     f"where FP wins {np.median(kap[~win]) if (~win).any() else float('nan'):.2f}")
    text = "\n".join(lines)
    open(os.path.join(out_dir, "storage_summary.md"), "w").write(text + "\n")
    print(text)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    width = 0.8 / len(rots)
    def sqnr(n, r):
        return -10 * math.log10(np.mean([float(x["nmse"]) for x in rows
                                         if x["format"] == n and int(x["rotation"]) == r]))
    base = sqnr("nvfp4", 0) if "nvfp4" in names else 0.0
    for j, r in enumerate(rots):
        vals = [sqnr(n, r) - base for n in names]
        a1.bar(np.arange(len(names)) + j * width, vals, width, label=f"rotation {r or 'none'}",
               color=["#2a78d6", "#eb6834", "#1baf7a"][j % 3])
    a1.axhline(0, color="#222222", linewidth=0.8)
    a1.set(xticks=np.arange(len(names)) + width * (len(rots) - 1) / 2, xticklabels=names,
           ylabel="SQNR vs NVFP4 without rotation (dB)", title="Quantization quality per format (higher is better)")
    a1.legend(frameon=False, fontsize=8)
    for i, n in enumerate(names):
        pts = [x for x in rows if x["format"] == n and int(x["rotation"]) == 0]
        a2.scatter([float(x["kappa_sq"]) for x in pts], [float(x["nmse"]) for x in pts], s=14,
                   color=colors[i % len(colors)], label=n)
    a2.set(xscale="log", yscale="log", xlabel="crest factor kappa^2 (format's block size)",
           ylabel="NMSE (lower is better)", title="Error vs crest factor, per tensor (no rotation)")
    a2.legend(frameon=False, fontsize=8)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "storage.png"), dpi=150)
    print("wrote", os.path.join(out_dir, "storage.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--gpt2", default="", help="path to GPT-2 model.safetensors (optional)")
    p.add_argument("--formats", nargs="+", default=list(F.FORMATS))
    p.add_argument("--rot-sizes", nargs="*", type=int, default=[16, 128])
    p.add_argument("--ptq", action="store_true", help="also evaluate models with quantized weights")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out-dir", default="results/storage")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fmts = [F.FORMATS[n] for n in args.formats]

    rows = []
    for path in args.checkpoints:
        meta, arrays = load_checkpoint(path)
        for name, w in arrays.items():
            if is_matmul_weight(name, w):
                rows += tensor_rows(f"{meta['runner']}:{os.path.basename(path)}", name, w, fmts, args.rot_sizes)
    if args.gpt2:
        weights = read_safetensors(args.gpt2, keep=lambda n, s: n.endswith(GPT2_KEEP))
        for name, w in weights.items():
            rows += tensor_rows("gpt2", name, w, fmts, args.rot_sizes)
            print(f"  gpt2 {name} {w.shape}", flush=True)
    with open(os.path.join(args.out_dir, "storage.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    summarize(rows, args.out_dir)

    if args.ptq:
        device = getattr(ndl, args.device)()
        ptq = []
        for path in args.checkpoints:
            ptq += ptq_rows(path, fmts, args.rot_sizes, device)
        with open(os.path.join(args.out_dir, "ptq.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(ptq[0]))
            wr.writeheader()
            wr.writerows(ptq)
        print("wrote", os.path.join(args.out_dir, "ptq.csv"))


if __name__ == "__main__":
    main()
