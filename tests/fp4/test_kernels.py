# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Tests for Stage 4: packed NVFP4 storage and the quantize+pack / dequantize /
GEMV kernels, on the cpu backend and (when built) the cuda backend.

Every kernel must match the NumPy reference *exactly*: same bytes for the packed
codes and E4M3 scales, same tensor scale, same float32 bits for dequantized
values and for the GEMV result (reference.nvfp4_gemv adds in the kernel's order).

    python3 -m pytest tests/fp4/test_kernels.py -v
"""
import sys

sys.path.append("./python")

import numpy as np
import pytest

from needle import backend_ndarray as nd
from needle.fp4 import reference as ref
from needle.fp4.packed import NVFP4Matrix

DEVICES = [
    nd.cpu(),
    pytest.param(nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="no CUDA")),
]

# Odd sizes on purpose: tails shorter than a block, rows whose padded length
# (48) is not a multiple of a warp's 32 elements, a single column, many rows.
SHAPES = [(1, 16), (3, 37), (8, 5), (1, 1), (64, 784), (33, 1000), (300, 257), (128, 4096)]


def _matrix(shape, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(np.float32)
    a.flat[::13] *= 50.0          # outliers so block scales cover a wide range
    return a


def _arr(x, device):
    return nd.array(np.asarray(x, dtype=np.float32), device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("stochastic", [False, True])
def test_quantize_pack_matches_reference(device, shape, stochastic):
    a = _matrix(shape, seed=1)
    rand = np.random.default_rng(2).random(shape, dtype=np.float32) if stochastic else None
    expected = ref.quantize_nvfp4(a, axis=1, stochastic=stochastic, rand=rand)
    W = NVFP4Matrix.quantize(_arr(a, device), stochastic=stochastic,
                             rand=None if rand is None else _arr(rand, device))
    got = W.to_reference()
    assert got.tensor_scale == expected.tensor_scale
    np.testing.assert_array_equal(got.packed, expected.packed)              # code bytes
    np.testing.assert_array_equal(ref.encode_e4m3(got.block_scales),       # scale bytes
                                  ref.encode_e4m3(expected.block_scales))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
def test_dequantize_matches_reference(device, shape):
    a = _matrix(shape, seed=3)
    W = NVFP4Matrix.quantize(_arr(a, device))
    np.testing.assert_array_equal(W.dequantize().numpy(), ref.fake_quant(a, axis=1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
def test_gemv_matches_reference_exactly(device, shape):
    a = _matrix(shape, seed=4)
    x = np.random.default_rng(5).standard_normal(shape[1]).astype(np.float32)
    W = NVFP4Matrix.quantize(_arr(a, device))
    y = W.gemv(_arr(x, device)).numpy()
    np.testing.assert_array_equal(y, ref.nvfp4_gemv(ref.quantize_nvfp4(a, axis=1), x))
    # Sanity: it is a matrix-vector product (same answer up to rounding in float64).
    np.testing.assert_allclose(y, ref.fake_quant(a, axis=1).astype(np.float64) @ x,
                               rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_all_zero_matrix(device):
    W = NVFP4Matrix.quantize(_arr(np.zeros((4, 20)), device))
    assert W.tensor_scale == 1.0
    np.testing.assert_array_equal(W.dequantize().numpy(), 0.0)
    np.testing.assert_array_equal(W.gemv(_arr(np.ones(20), device)).numpy(), 0.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(5, 7), (64, 784), (300, 256)])
def test_fp32_gemv_baseline(device, shape):
    w = _matrix(shape, seed=6)
    x = np.random.default_rng(7).standard_normal(shape[1]).astype(np.float32)
    y = nd.NDArray.make((shape[0],), device=device)
    device.gemv_fp32(_arr(w, device)._handle, _arr(x, device)._handle, y._handle, *shape)
    np.testing.assert_allclose(y.numpy(), w.astype(np.float64) @ x, rtol=1e-4, atol=1e-3)


def test_memory_accounting():
    W = NVFP4Matrix.quantize(_arr(_matrix((4096, 4096), seed=8), nd.cpu()))
    # 4096*4096 codes / 2 per byte + one scale byte per 16 values + 4-byte tensor scale
    assert W.nbytes == 4096 * 4096 // 2 + 4096 * 4096 // 16 + 4
    assert W.dense_nbytes == 4096 * 4096 * 4
    assert 7.0 < W.dense_nbytes / W.nbytes < 7.2      # about 7.1x smaller


def test_numpy_backend_is_rejected():
    with pytest.raises(NotImplementedError):
        NVFP4Matrix.quantize(_arr(np.ones((2, 16)), nd.cpu_numpy()))
