# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""NVFP4 vs MXFP4 weight quantization (NumPy reference), for the stretch-goal comparison.

Both use the same E2M1 grid and quantize a 2-D array in blocks along its last axis;
they differ only in the block size and the scale:
    NVFP4: 16 elements, E4M3 block scale times an FP32 tensor scale (4.5 bits per weight)
    MXFP4: 32 elements, power-of-two (E8M0) block scale            (4.25 bits per weight)
"""
import numpy as np

from . import reference as ref

BITS = {"nvfp4": 4 + 8 / 16, "mxfp4": 4 + 8 / 32}


def quantize_nvfp4(w):
    return ref.fake_quant(np.asarray(w, dtype=np.float32), axis=1)


def quantize_mxfp4(w):
    """OCP MX rule: scale = 2^(floor(log2(block amax)) - 2), so amax / scale lies in [4, 8)."""
    w = np.asarray(w, dtype=np.float32)
    pad = (-w.shape[1]) % 32
    b = np.pad(w, ((0, 0), (0, pad))).reshape(w.shape[0], -1, 32)
    amax = np.abs(b).max(-1, keepdims=True)
    s = np.exp2(np.floor(np.log2(np.maximum(amax, 1e-30))) - 2).astype(np.float32)
    q = (ref.e2m1_decode(ref.e2m1_encode(b / s)) * s).reshape(w.shape[0], -1)
    return q[:, : w.shape[1]]


QUANTIZERS = {"nvfp4": quantize_nvfp4, "mxfp4": quantize_mxfp4}


def nmse(q, w):
    """Normalized squared error ||q - w||^2 / ||w||^2; SQNR in dB is -10 log10 of it."""
    w = np.asarray(w, np.float64)
    return float(np.sum((np.asarray(q, np.float64) - w) ** 2) / np.sum(w ** 2))
