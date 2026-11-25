# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Stage 4 benchmark: FP32 GEMV vs NVFP4 GEMV (y = W x), the core operation of
text generation at batch size 1.

For each matrix size it reports:
    time_us          average kernel time (after warm-up; CUDA events on the GPU)
    weight_MiB       memory used by W
    bytes_moved_MiB  bytes the kernel must read/write at least once: W + x + y
    bandwidth_GB_s   bytes_moved / time = how fast the kernel streams memory
    pct_of_peak      bandwidth as a percentage of the GPU's peak DRAM bandwidth
    speedup          FP32 time / this time
    rel_error        ||y - y_fp32|| / ||y_fp32||, the accuracy cost of 4-bit weights

    python3 benchmarks/bench_gemv.py --device cuda                 # 4096 x 4096
    python3 benchmarks/bench_gemv.py --device cuda --sizes 1024 2048 4096 8192

On the CPU it uses wall-clock timing (only to check the script works; the
numbers that matter are GPU numbers).
"""
import argparse
import csv
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

import needle as ndl
from needle import backend_ndarray as nd
from needle.fp4.packed import NVFP4Matrix

MiB = 2 ** 20


def time_cpu(fn, warmup, iters):
    """Average wall-clock milliseconds per call."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters * 1e3


def bench_size(device, n, warmup, iters):
    rng = np.random.default_rng(n)
    w_np = rng.standard_normal((n, n)).astype(np.float32)
    x_np = rng.standard_normal(n).astype(np.float32)
    w = nd.array(w_np, device=device)
    x = nd.array(x_np, device=device)
    y32 = nd.NDArray.make((n,), device=device)
    W4 = NVFP4Matrix.quantize(w)
    y4 = nd.NDArray.make((n,), device=device)
    on_gpu = device.name == "cuda"

    # --- FP32 GEMV: our warp-per-row kernel -----------------------------------
    if on_gpu:
        t32 = device.bench_gemv_fp32(w._handle, x._handle, y32._handle, n, n, warmup, iters)
    else:
        t32 = time_cpu(lambda: device.gemv_fp32(w._handle, x._handle, y32._handle, n, n),
                       warmup, iters)
    # --- NVFP4 GEMV: reads the packed 4-bit weights directly -------------------
    if on_gpu:
        t4 = device.bench_nvfp4_gemv(W4.packed, W4.scales, W4.tensor_scale, x._handle,
                                     y4._handle, n, n, warmup, iters)
    else:
        t4 = time_cpu(lambda: device.nvfp4_gemv(W4.packed, W4.scales, W4.tensor_scale,
                                                x._handle, y4._handle, n, n), warmup, iters)
    device.gemv_fp32(w._handle, x._handle, y32._handle, n, n)
    device.nvfp4_gemv(W4.packed, W4.scales, W4.tensor_scale, x._handle, y4._handle, n, n)

    # --- the homework's general matmul as a third reference point --------------
    x_col = x.reshape((n, 1))
    def homework_matmul():
        out = w @ x_col
        if on_gpu:
            device.synchronize()
        return out
    t_hw = time_cpu(homework_matmul, max(1, warmup // 10), max(3, iters // 10))

    y_ref = y32.numpy().astype(np.float64)
    rel_err = np.linalg.norm(y4.numpy() - y_ref) / np.linalg.norm(y_ref)
    vec_bytes = 2 * n * 4                       # read x, write y
    rows = []
    for name, t_ms, w_bytes, err in [
        ("fp32 gemv (ours)", t32, n * n * 4, 0.0),
        ("nvfp4 gemv (ours)", t4, W4.nbytes, rel_err),
        ("fp32 needle matmul (hw)", t_hw, n * n * 4, 0.0),
    ]:
        moved = w_bytes + vec_bytes
        rows.append({"n": n, "kernel": name, "time_us": t_ms * 1e3,
                     "weight_MiB": w_bytes / MiB, "bytes_moved_MiB": moved / MiB,
                     "bandwidth_GB_s": moved / (t_ms * 1e-3) / 1e9,
                     "speedup": t32 / t_ms, "rel_error": err})
    return rows


def plot(rows, path, peak):
    """Two panels per matrix size: kernel time, and achieved bandwidth (with the GPU's peak)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kernels = [("fp32 gemv (ours)", "#2a78d6", "FP32 GEMV"), ("nvfp4 gemv (ours)", "#eb6834", "NVFP4 GEMV")]
    sizes = sorted({r["n"] for r in rows})
    x = np.arange(len(sizes))
    fig, (ax_t, ax_bw) = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, (name, color, label) in enumerate(kernels):
        t = [next(r["time_us"] for r in rows if r["n"] == n and r["kernel"] == name) for n in sizes]
        bw = [next(r["bandwidth_GB_s"] for r in rows if r["n"] == n and r["kernel"] == name) for n in sizes]
        ax_t.bar(x + (k - 0.5) * 0.38, t, 0.36, color=color, label=label)
        ax_bw.bar(x + (k - 0.5) * 0.38, bw, 0.36, color=color, label=label)
        for xi, ti in zip(x, t):
            ax_t.annotate(f"{ti:.0f}", (xi + (k - 0.5) * 0.38, ti), ha="center", va="bottom", fontsize=8)
    if peak:
        ax_bw.axhline(peak, color="#555555", linestyle="--", linewidth=1)
        ax_bw.annotate(f"GPU peak {peak:.0f} GB/s", (x[-1] + 0.4, peak), ha="right", va="bottom", fontsize=8)
    labels = [f"{n}x{n}" for n in sizes]
    ax_t.set(xticks=x, xticklabels=labels, ylabel="time per GEMV (microseconds)", yscale="log",
             title="Kernel time (lower is better)")
    ax_bw.set(xticks=x, xticklabels=labels, ylabel="achieved bandwidth (GB/s)",
              title="Memory bandwidth used (higher is better)")
    for ax in (ax_t, ax_bw):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("y = W x with W stored in FP32 vs NVFP4 (4-bit codes + E4M3 block scales)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--out", default="results/bench_gemv.csv")
    args = parser.parse_args()

    device = getattr(ndl, args.device)()
    if not device.enabled():
        parser.error(f"needle's {args.device} backend is not built (Colab: use a GPU runtime)")
    peak = None
    if args.device == "cuda":
        info = device.device_info()
        peak = info["peak_bandwidth_gb_s"]
        print(f"GPU: {info['name']}, peak DRAM bandwidth {peak:.0f} GB/s, "
              f"{info['memory_gb']:.1f} GB memory")

    rows = []
    for n in args.sizes:
        rows += bench_size(device, n, args.warmup, args.iters)
    for r in rows:
        r["pct_of_peak"] = 100 * r["bandwidth_GB_s"] / peak if peak else float("nan")

    header = f"{'n':>6} {'kernel':24s} {'time_us':>10} {'weight_MiB':>10} {'moved_MiB':>10} " \
             f"{'GB/s':>8} {'%peak':>6} {'speedup':>8} {'rel_err':>9}"
    print(header)
    for r in rows:
        print(f"{r['n']:>6} {r['kernel']:24s} {r['time_us']:>10.1f} {r['weight_MiB']:>10.2f} "
              f"{r['bytes_moved_MiB']:>10.2f} {r['bandwidth_GB_s']:>8.1f} {r['pct_of_peak']:>6.1f} "
              f"{r['speedup']:>7.2f}x {r['rel_error']:>9.2e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out)
    plot(rows, os.path.splitext(args.out)[0] + ".png", peak)


if __name__ == "__main__":
    main()
