"""Plug-in 4-bit block formats (NumPy reference) for the storage study.

Each format quantizes a 2-D array in blocks along its last axis and returns the
dequantized array. All share one interface so they can be swapped freely, with
an optional random Hadamard rotation along the same axis.
"""
from dataclasses import dataclass

import numpy as np

from . import reference as ref

INT4_MAX = 7
# NF4 levels from QLoRA (Dettmers et al., 2023), as in bitsandbytes.
NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class BlockFormat:
    name: str
    grid: str          # "e2m1" | "int4" | "nf4"
    block: int
    scale: str         # "e4m3+fp32" | "e8m0" | "fp16"

    @property
    def bits(self) -> float:
        scale_bits = {"e4m3+fp32": 8, "e8m0": 8, "fp16": 16}[self.scale]
        return 4 + scale_bits / self.block


FORMATS = {f.name: f for f in [
    BlockFormat("nvfp4", "e2m1", 16, "e4m3+fp32"),
    BlockFormat("nvint4", "int4", 16, "e4m3+fp32"),
    BlockFormat("mxfp4", "e2m1", 32, "e8m0"),
    BlockFormat("int4_g32", "int4", 32, "fp16"),
    BlockFormat("int4_g128", "int4", 128, "fp16"),
    BlockFormat("nf4_g64", "nf4", 64, "fp16"),
]}


def _blocks(w, g):
    pad = (-w.shape[1]) % g
    w = np.pad(w, ((0, 0), (0, pad)))
    return w.reshape(w.shape[0], -1, g), pad


def _unblocks(b, pad):
    w = b.reshape(b.shape[0], -1)
    return w[:, : w.shape[1] - pad] if pad else w


def _round_grid(v, levels):
    idx = np.abs(v[..., None] - levels).argmin(-1)
    return levels[idx]


def _e2m1(v):
    return ref.e2m1_decode(ref.e2m1_encode(v))


def quantize(w, fmt: BlockFormat):
    """Quantize-dequantize a 2-D float array with `fmt`, blocks along axis 1."""
    w = np.asarray(w, dtype=np.float32)
    if fmt.name == "nvfp4":
        return ref.fake_quant(w, axis=1)
    b, pad = _blocks(w, fmt.block)
    amax = np.abs(b).max(-1, keepdims=True)

    if fmt.grid == "e2m1":                                   # MXFP4
        s = np.exp2(np.floor(np.log2(np.maximum(amax, 1e-30))) - 2).astype(np.float32)
        q = _e2m1(b / s) * s
    elif fmt.grid == "int4":
        if fmt.scale == "e4m3+fp32":                         # NVINT4
            ts = np.float32(np.abs(w).max() / (INT4_MAX * float(ref.E4M3_MAX))) or np.float32(1)
            s = ref.round_to_e4m3(amax / INT4_MAX / ts) * ts
        else:
            s = (amax / INT4_MAX).astype(np.float16).astype(np.float32)
        safe = np.where(s > 0, s, 1)
        q = np.clip(np.round(b / safe), -INT4_MAX, INT4_MAX) * s
    else:                                                    # NF4
        s = amax.astype(np.float16).astype(np.float32)
        safe = np.where(s > 0, s, 1)
        q = _round_grid(b / safe, NF4_LEVELS) * s
    return _unblocks(q.astype(np.float32), pad)


def hadamard(n):
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


class Rotation:
    """Random-sign Hadamard rotation in blocks of `size` along axis 1: w -> (w * signs) @ H."""

    def __init__(self, cols, size, seed=0):
        assert size & (size - 1) == 0, "rotation size must be a power of two"
        self.size, self.h = size, hadamard(size)
        self.signs = np.random.default_rng(seed).choice([-1.0, 1.0], cols).astype(np.float32)

    def forward(self, w):
        b, pad = _blocks(w * self.signs, self.size)
        return _unblocks(b @ self.h, pad)

    def inverse(self, w):
        b, pad = _blocks(w, self.size)
        return _unblocks(b @ self.h.T, pad) * self.signs


def quantize_rotated(w, fmt, rot_size, seed=0):
    rot = Rotation(w.shape[1], rot_size, seed)
    return rot.inverse(quantize(rot.forward(w), fmt))


def crest_factor_sq(w, g):
    """kappa^2 = E[max over block of w^2] / E[w^2]."""
    b, _ = _blocks(np.asarray(w, dtype=np.float64), g)
    return float(np.mean(np.abs(b).max(-1) ** 2) / np.mean(np.asarray(w, np.float64) ** 2))


def nmse(q, w):
    w = np.asarray(w, np.float64)
    return float(np.sum((np.asarray(q, np.float64) - w) ** 2) / np.sum(w ** 2))


def predicted_nmse_int4(kappa_sq):
    """First-order model for symmetric INT4: uniform error step^2/12 with step = amax/7."""
    return kappa_sq / (12 * INT4_MAX ** 2)
