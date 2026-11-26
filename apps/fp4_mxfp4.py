# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Stretch goal: NVFP4 vs MXFP4 on trained weights.

Takes FP32 checkpoints saved by fp4_experiment.py --save-weights and, for each format,
  1. quantizes every matmul weight and reports its SQNR (dB, higher is better),
  2. loads the quantized weights back into the model and reports test accuracy.

    python3 apps/fp4_mxfp4.py results/ckpt/mlp_fp32_s0.npz --out results/mxfp4.md
"""
import argparse
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.append(os.path.dirname(__file__))

import needle as ndl
from needle.fp4 import formats as F
from fp4_common import assign_weights, load_checkpoint
from fp4_experiment import ptq_target


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", default="results/mxfp4.md")
    args = p.parse_args()
    device = getattr(ndl, args.device)()

    lines = ["| checkpoint | format | bits per weight | weight SQNR (dB) | test accuracy (%) |",
             "|---|---|---|---|---|"]
    for path in args.checkpoints:
        meta, arrays = load_checkpoint(path)
        model, evaluate = ptq_target(meta, device)
        assign_weights(model, arrays, device)
        name = os.path.basename(path)
        lines.append(f"| {name} | fp32 | 32 | | {100 * evaluate(model)['test_acc']:.2f} |")
        for fmt, quantize in F.QUANTIZERS.items():
            # needle stores W as (in, out) and reduces over `in`, so block along axis 0.
            q = {k: (quantize(v.T).T if v.ndim == 2 else v) for k, v in arrays.items()}
            total = sum(((q[k] - v) ** 2).sum() for k, v in arrays.items() if v.ndim == 2)
            power = sum((v ** 2).sum() for v in arrays.values() if v.ndim == 2)
            assign_weights(model, q, device)
            acc = evaluate(model)["test_acc"]
            lines.append(f"| {name} | {fmt} | {F.BITS[fmt]} | {-10 * math.log10(total / power):.2f} | "
                         f"{100 * acc:.2f} |")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
