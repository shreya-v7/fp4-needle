# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Tests for Stage 2: needle fake quantization, the FP4Matmul Op, and nn.QuantizedLinear.

Every needle result is compared with the NumPy reference (tests/fp4/test_reference.py
checks the reference itself). Run from the repo root:
    python3 -m pytest tests/fp4/test_fake_quant.py -v
"""
import sys

sys.path.append("./python")

import numpy as np
import pytest

import needle as ndl
from needle import backend_ndarray as nd
from needle.fp4 import quantize, reference as ref
from needle.fp4.config import FP32, FPROP, FPROP_DGRAD, FPROP_DGRAD_WGRAD, QuantConfig

# All needle devices; the CUDA one is skipped when needle was built without it.
DEVICES = [
    nd.cpu(),
    nd.cpu_numpy(),
    pytest.param(nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="no CUDA")),
]


def _arr(x, device):
    return nd.array(np.asarray(x, dtype=np.float32), device=device)


# ----------------------------------------------------------------------------
# New backend primitives: floor and abs
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_floor_and_abs_primitives(device):
    x = np.array([-2.5, -1.0, -0.5, -0.0, 0.0, 0.25, 1.0, 1.5, 3.99, 1e6], dtype=np.float32)
    np.testing.assert_array_equal(_arr(x, device).floor().numpy(), np.floor(x))
    np.testing.assert_array_equal(_arr(x, device).abs().numpy(), np.abs(x))
    # Non-compact input (a transposed view) must be compacted first.
    y = np.arange(-6, 6, dtype=np.float32).reshape(3, 4) / 4
    view = _arr(y, device).permute((1, 0))
    np.testing.assert_array_equal(view.floor().numpy(), np.floor(y.T))


# ----------------------------------------------------------------------------
# Building blocks match the reference bit for bit
# ----------------------------------------------------------------------------

def _e4m3_test_values():
    """Values that stress E4M3 rounding: every E4M3 value, every midpoint between
    neighbours (ties), values one float32 ulp either side of powers of two,
    random values across the whole range, and values above 448."""
    grid = np.unique(ref.decode_e4m3(np.arange(0, 0x7F, dtype=np.uint8)))
    midpoints = (grid[:-1] + grid[1:]) / 2
    powers = np.float32(2.0) ** np.arange(-12, 10, dtype=np.float32)
    near_powers = np.concatenate([np.nextafter(powers, 0), powers, np.nextafter(powers, np.inf)])
    random = np.exp(np.random.default_rng(0).uniform(np.log(1e-4), np.log(600), 2000))
    return np.concatenate([grid, midpoints, near_powers, random, [449, 464, 480, 1e5]]).astype(np.float32)


@pytest.mark.parametrize("device", DEVICES)
def test_round_to_e4m3_matches_reference(device):
    v = _e4m3_test_values()
    np.testing.assert_array_equal(quantize.round_to_e4m3(_arr(v, device)).numpy(), ref.round_to_e4m3(v))


@pytest.mark.parametrize("device", DEVICES)
def test_round_to_e2m1_matches_reference(device):
    # A dense sweep over [-7, 7] that includes every grid value and every tie.
    v = np.concatenate([np.linspace(-7, 7, 2801), ref.E2M1_GRID, [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]])
    v = v.astype(np.float32)
    expected = ref.e2m1_decode(ref.e2m1_encode(v))
    np.testing.assert_array_equal(quantize.round_to_e2m1(_arr(v, device)).numpy(), expected)

    # Stochastic rounding with the same random numbers must give the same values.
    rand = np.random.default_rng(1).random(v.shape, dtype=np.float32)
    expected_sr = ref.e2m1_decode(ref.e2m1_encode(v, stochastic=True, rand=rand))
    out_sr = quantize.round_to_e2m1(_arr(v, device), rand=_arr(rand, device)).numpy()
    np.testing.assert_array_equal(out_sr, expected_sr)


# ----------------------------------------------------------------------------
# Full fake quantization matches reference.fake_quant bit for bit
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape, axis", [
    ((16, 16), 1), ((16, 16), 0),
    ((5, 37), 1), ((37, 5), 0),        # padded tails
    ((64, 784), 1), ((784, 64), 0),    # MNIST layer sizes
    ((3, 1), 0), ((2, 4, 33), 1),      # tiny and 3-D
])
@pytest.mark.parametrize("stochastic", [False, True])
def test_fake_quant_matches_reference(device, shape, axis, stochastic):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(shape).astype(np.float32)
    x.flat[::7] *= 30.0   # outliers, so block scales span a wide range
    rand = rng.random(shape, dtype=np.float32) if stochastic else None
    expected = ref.fake_quant(x, axis=axis, stochastic=stochastic, rand=rand)
    out = quantize.fake_quant(_arr(x, device), axis=axis, stochastic=stochastic,
                              rand=None if rand is None else _arr(rand, device))
    assert out.shape == x.shape
    np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_fake_quant_edge_cases(device):
    zeros = np.zeros((4, 20), dtype=np.float32)
    np.testing.assert_array_equal(quantize.fake_quant(_arr(zeros, device), axis=1).numpy(), zeros)

    tiny = np.full((1, 32), 1e-3, dtype=np.float32)   # block 1 underflows E4M3 (see reference tests)
    tiny[0, 0] = 1e6
    out = quantize.fake_quant(_arr(tiny, device), axis=1).numpy()
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, ref.fake_quant(tiny, axis=1))


# ----------------------------------------------------------------------------
# FP4Matmul Op: forward and backward
# ----------------------------------------------------------------------------

def _run_matmul(config, X, W, G, device, seed=0):
    """Forward Y = fp4_matmul(X, W), then backward with dL/dY = G.

    loss = sum(Y * G) makes the incoming gradient of Y exactly G. The seed is
    set just before backward because SR draws its random numbers there."""
    Xt = ndl.Tensor(_arr(X, device), device=device)
    Wt = ndl.Tensor(_arr(W, device), device=device)
    Y = ndl.ops.fp4_matmul(Xt, Wt, config)
    loss = (Y * ndl.Tensor(_arr(G, device), device=device)).sum()
    np.random.seed(seed)
    loss.backward()
    return Y.numpy(), Xt.grad.numpy(), Wt.grad.numpy()


def _expected(config, X, W, G, device, seed=0):
    """Same three matmuls, with operands fake-quantized by the NumPy reference and
    multiplied with needle's own matmul (so float32 summation order is identical)."""
    fq = ref.fake_quant
    mm = lambda a, b: (_arr(a, device) @ _arr(b, device)).numpy()

    Y = mm(fq(X, 1), fq(W, 0)) if config.fprop else mm(X, W)

    # Replay the random draws in the Op's order: Dgrad's dY first, then Wgrad's.
    np.random.seed(seed)
    draw = lambda shape: np.random.rand(*shape).astype(np.float32)
    if config.dgrad:
        rand = draw(G.shape) if config.stochastic else None
        dX = mm(fq(G, 1, config.stochastic, rand), fq(W, 1).T)
    else:
        dX = mm(G, W.T)
    if config.wgrad:
        rand = draw(G.shape) if config.stochastic else None
        dW = mm(fq(X, 0).T, fq(G, 0, config.stochastic, rand))
    else:
        dW = mm(X.T, G)
    return Y, dX, dW


ALL_CONFIGS = [FP32, FPROP, FPROP_DGRAD, FPROP_DGRAD_WGRAD,
               QuantConfig(fprop=True, dgrad=True, stochastic=True),
               QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True),
               QuantConfig(wgrad=True), QuantConfig(dgrad=True, stochastic=True)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
@pytest.mark.parametrize("batch, n_in, n_out", [(32, 48, 16), (13, 37, 21)])
def test_fp4_matmul_matches_reference(device, config, batch, n_in, n_out):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((batch, n_in)).astype(np.float32)
    W = rng.standard_normal((n_in, n_out)).astype(np.float32)
    G = rng.standard_normal((batch, n_out)).astype(np.float32)
    for got, want in zip(_run_matmul(config, X, W, G, device), _expected(config, X, W, G, device)):
        np.testing.assert_array_equal(got, want)


# ----------------------------------------------------------------------------
# nn.QuantizedLinear
# ----------------------------------------------------------------------------

def _linear_outputs(layer, X, device):
    Xt = ndl.Tensor(_arr(X, device), device=device)
    out = layer(Xt)
    (out * out).sum().backward()  # any scalar loss works for this comparison
    return out.numpy(), Xt.grad.numpy(), layer.weight.grad.numpy(), layer.bias.grad.numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_quantized_linear_all_off_equals_linear_exactly(device):
    np.random.seed(4)   # needle's init draws from np.random, so equal seeds give equal weights
    linear = ndl.nn.Linear(37, 21, device=device)
    np.random.seed(4)
    qlinear = ndl.nn.QuantizedLinear(37, 21, device=device, config=FP32)
    np.testing.assert_array_equal(linear.weight.numpy(), qlinear.weight.numpy())

    X = np.random.default_rng(5).standard_normal((13, 37)).astype(np.float32)
    for a, b in zip(_linear_outputs(linear, X, device), _linear_outputs(qlinear, X, device)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_quantized_linear_forward_matches_reference(device):
    np.random.seed(6)
    layer = ndl.nn.QuantizedLinear(40, 24, device=device, config=FPROP)
    X = np.random.default_rng(7).standard_normal((8, 40)).astype(np.float32)
    W, b = layer.weight.numpy(), layer.bias.numpy()
    expected = (_arr(ref.fake_quant(X, 1), device) @ _arr(ref.fake_quant(W, 0), device)).numpy() + b
    got = layer(ndl.Tensor(_arr(X, device), device=device)).numpy()
    np.testing.assert_array_equal(got, expected)


def test_quantized_mlp_trains():
    """Smoke test: a small MLP using QuantizedLinear with everything in FP4 + SR
    still learns a simple synthetic classification task."""
    device = nd.cpu()
    rng = np.random.default_rng(8)
    X = rng.standard_normal((128, 32)).astype(np.float32)
    y = (X[:, :4].sum(axis=1) > 0).astype(np.uint8)
    np.random.seed(9)
    cfg = QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True)
    model = ndl.nn.Sequential(
        ndl.nn.QuantizedLinear(32, 64, device=device, config=cfg), ndl.nn.ReLU(),
        ndl.nn.QuantizedLinear(64, 2, device=device, config=cfg),
    )
    opt = ndl.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = ndl.nn.SoftmaxLoss()
    Xt, yt = ndl.Tensor(_arr(X, device), device=device), ndl.Tensor(y, device=device)
    losses = []
    for _ in range(40):
        opt.reset_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.numpy()))
    assert losses[-1] < 0.5 * losses[0]
