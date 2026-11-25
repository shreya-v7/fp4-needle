# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Tests for the plug-in formats, rotation, monitor and checkpoints used by the paper experiments."""
import os
import sys

sys.path.append("./python")
sys.path.append("./apps")

import numpy as np
import pytest

import needle as ndl
from needle import backend_ndarray as nd
from needle.fp4 import formats as F, reference as ref, monitor
from needle.fp4.config import QuantConfig

RNG = np.random.default_rng(0)


def test_bits_per_weight():
    assert F.FORMATS["nvfp4"].bits == 4.5 and F.FORMATS["mxfp4"].bits == 4.25
    assert F.FORMATS["int4_g128"].bits == 4.125 and F.FORMATS["nf4_g64"].bits == 4.25


def test_nvfp4_matches_reference():
    w = RNG.standard_normal((8, 50)).astype(np.float32)
    np.testing.assert_array_equal(F.quantize(w, F.FORMATS["nvfp4"]), ref.fake_quant(w, axis=1))


@pytest.mark.parametrize("name", ["nvint4", "int4_g32", "int4_g128"])
def test_int4_uses_15_levels_per_block(name):
    fmt = F.FORMATS[name]
    w = RNG.standard_normal((4, 256)).astype(np.float32)
    q = F.quantize(w, fmt)
    for blk in q.reshape(4, -1, fmt.block):
        for row in blk:
            assert len(np.unique(np.round(row / np.abs(row).max() * 7, 4))) <= 15


def test_mxfp4_is_e2m1_times_power_of_two_scale():
    w = RNG.standard_normal((2, 64)).astype(np.float32) * 3
    q = F.quantize(w, F.FORMATS["mxfp4"]).reshape(2, -1, 32)
    scale = np.exp2(np.floor(np.log2(np.abs(w.reshape(2, -1, 32)).max(-1, keepdims=True))) - 2)
    v = np.abs(q / scale)
    assert np.all(np.min(np.abs(v[..., None] - ref.E2M1_GRID), -1) == 0)


def test_nf4_values_on_grid():
    w = RNG.standard_normal((3, 128)).astype(np.float32)
    q = F.quantize(w, F.FORMATS["nf4_g64"]).reshape(3, 2, 64)
    s = np.abs(w.reshape(3, 2, 64)).max(-1, keepdims=True).astype(np.float16).astype(np.float32)
    levels = q / s
    assert np.all(np.min(np.abs(levels[..., None] - F.NF4_LEVELS), -1) < 1e-5)


@pytest.mark.parametrize("cols, size", [(64, 16), (64, 128), (100, 16), (37, 32)])
def test_rotation_round_trip(cols, size):
    w = RNG.standard_normal((5, cols)).astype(np.float32)
    rot = F.Rotation(cols, size, seed=1)
    np.testing.assert_allclose(rot.inverse(rot.forward(w)), w, atol=1e-5)


def test_rotation_lowers_crest_factor_of_outliers():
    w = RNG.standard_normal((64, 256)).astype(np.float32)
    w[:, :3] *= 40
    rot = F.Rotation(256, 16)
    assert F.crest_factor_sq(rot.forward(w), 16) < 0.5 * F.crest_factor_sq(w, 16)


def test_int4_error_model_on_gaussian():
    w = RNG.standard_normal((512, 512)).astype(np.float32)
    for name in ("int4_g32", "int4_g128"):
        g = F.FORMATS[name].block
        pred = F.predicted_nmse_int4(F.crest_factor_sq(w, g))
        meas = F.nmse(F.quantize(w, F.FORMATS[name]), w)
        assert abs(meas / pred - 1) < 0.1


def _mlp(config):
    from fp4_experiment import make_mlp
    np.random.seed(0)
    return make_mlp(config, 32, nd.cpu(), in_features=20)


def test_monitor_ratio():
    X = ndl.Tensor(RNG.standard_normal((16, 20)).astype(np.float32), device=nd.cpu())
    y = ndl.Tensor(RNG.integers(0, 10, 16).astype(np.float32), device=nd.cpu())
    loss = ndl.nn.SoftmaxLoss()
    exact = _mlp(QuantConfig(fprop=True))
    assert monitor.grad_noise_ratio(exact, lambda: loss(exact(X), y)) == float("inf")
    fp4 = _mlp(QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True))
    r = monitor.grad_noise_ratio(fp4, lambda: loss(fp4(X), y))
    assert 0 < r < float("inf")
    assert fp4.modules[1].config.wgrad, "monitor must restore the configs"


def test_checkpoint_round_trip(tmp_path):
    from fp4_common import save_checkpoint, load_checkpoint, assign_weights, named_parameters
    m = _mlp(QuantConfig())
    path = str(tmp_path / "m.npz")
    save_checkpoint(m, path, {"runner": "mlp"})
    meta, arrays = load_checkpoint(path)
    assert meta == {"runner": "mlp"} and len(arrays) == len(named_parameters(m))
    m2 = _mlp(QuantConfig())
    for _, p in named_parameters(m2):
        p.data = p * 0
    assign_weights(m2, arrays, nd.cpu())
    for (_, a), (_, b) in zip(named_parameters(m), named_parameters(m2)):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_safetensors_reader(tmp_path):
    import json, struct
    from fp4_common import read_safetensors
    a = RNG.standard_normal((3, 4)).astype(np.float32)
    header = json.dumps({"x.weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, a.nbytes]}}).encode()
    path = tmp_path / "t.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + a.tobytes())
    np.testing.assert_array_equal(read_safetensors(str(path))["x.weight"], a)


def test_transformer_config_none_equals_fp32_config():
    from fp4_gpt_experiment import MiniGPT
    x = ndl.Tensor(RNG.integers(0, 11, (8, 2)).astype(np.float32), device=nd.cpu())
    np.random.seed(3)
    a = MiniGPT(11, 16, 1, 2, 8, None, nd.cpu())
    np.random.seed(3)
    b = MiniGPT(11, 16, 1, 2, 8, QuantConfig(), nd.cpu())
    np.testing.assert_array_equal(a(x).numpy(), b(x).numpy())
