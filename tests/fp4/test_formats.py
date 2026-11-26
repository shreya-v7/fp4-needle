# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Tests for the NVFP4 vs MXFP4 comparison and the checkpoint helpers it uses."""
import sys

sys.path.append("./python")
sys.path.append("./apps")

import numpy as np

from needle import backend_ndarray as nd
from needle.fp4 import formats as F, reference as ref
from needle.fp4.config import QuantConfig

RNG = np.random.default_rng(0)


def test_bits_per_weight():
    assert F.BITS == {"nvfp4": 4.5, "mxfp4": 4.25}


def test_nvfp4_matches_reference():
    w = RNG.standard_normal((8, 50)).astype(np.float32)
    np.testing.assert_array_equal(F.quantize_nvfp4(w), ref.fake_quant(w, axis=1))


def test_mxfp4_is_e2m1_times_power_of_two_scale():
    w = RNG.standard_normal((2, 64)).astype(np.float32) * 3
    q = F.quantize_mxfp4(w).reshape(2, -1, 32)
    scale = np.exp2(np.floor(np.log2(np.abs(w.reshape(2, -1, 32)).max(-1, keepdims=True))) - 2)
    v = np.abs(q / scale)
    assert np.all(np.min(np.abs(v[..., None] - ref.E2M1_GRID), -1) == 0)


def test_mxfp4_pads_and_crops_tail():
    w = RNG.standard_normal((3, 45)).astype(np.float32)
    assert F.quantize_mxfp4(w).shape == (3, 45)


def test_nvfp4_beats_mxfp4_on_gaussian_weights():
    w = RNG.standard_normal((256, 256)).astype(np.float32)
    assert F.nmse(F.quantize_nvfp4(w), w) < F.nmse(F.quantize_mxfp4(w), w)


def test_checkpoint_round_trip(tmp_path):
    from fp4_common import save_checkpoint, load_checkpoint, assign_weights, named_parameters
    from fp4_experiment import make_mlp
    np.random.seed(0)
    m = make_mlp(QuantConfig(), 32, nd.cpu(), in_features=20)
    path = str(tmp_path / "m.npz")
    save_checkpoint(m, path, {"runner": "mlp"})
    meta, arrays = load_checkpoint(path)
    assert meta == {"runner": "mlp"} and len(arrays) == len(named_parameters(m))
    m2 = make_mlp(QuantConfig(), 32, nd.cpu(), in_features=20)
    for _, p in named_parameters(m2):
        p.data = p * 0
    assign_weights(m2, arrays, nd.cpu())
    for (_, a), (_, b) in zip(named_parameters(m), named_parameters(m2)):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_every_config_starts_from_the_same_weights():
    """Bias audit: with the same seed, FP32 and FP4 models get identical initial weights,
    so differences between runs come only from FP4."""
    from fp4_experiment import EXTENDED, make_mlp
    from fp4_common import named_parameters
    weights = []
    for config in EXTENDED:
        np.random.seed(3)
        model = make_mlp(config, 16, nd.cpu(), in_features=20, depth=3)
        weights.append([p.numpy() for _, p in named_parameters(model)])
    for w in weights[1:]:
        for a, b in zip(weights[0], w):
            np.testing.assert_array_equal(a, b)
