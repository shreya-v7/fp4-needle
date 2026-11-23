"""Tests for the NumPy NVFP4 reference quantizer (python/needle/fp4/reference.py).

Run from the repo root:  python3 -m pytest tests/fp4/test_reference.py -v
"""
import sys

sys.path.append("./python")

import numpy as np
import pytest

from needle.fp4 import reference as ref


# ----------------------------------------------------------------------------
# FP8 E4M3 block-scale rounding
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (448.0, 448.0),      # largest E4M3 value
        (500.0, 448.0),      # saturates (no inf in E4M3)
        (1e6, 448.0),
        (1.0, 1.0),
        # 0.3 is in the binade [0.25, 0.5), spacing 2**-5 = 0.03125:
        # 0.3 / 0.03125 = 9.6 -> 10 -> 0.3125
        (0.3, 0.3125),
        # 17 in [16, 32), spacing 2: 17/2 = 8.5 is a tie -> even 8 -> 16
        (17.0, 16.0),
        # 19/2 = 9.5 tie -> even 10 -> 20
        (19.0, 20.0),
        (2.0 ** -6, 2.0 ** -6),   # smallest normal
        (2.0 ** -9, 2.0 ** -9),   # smallest subnormal (spacing below 2**-6 is 2**-9)
        (3 * 2.0 ** -10, 2.0 ** -8),  # 1.5 subnormal steps: tie -> even 2 steps
        (2.0 ** -10, 0.0),        # 0.5 subnormal steps: tie -> even 0
        (0.0, 0.0),
    ],
)
def test_round_to_e4m3_hand_examples(value, expected):
    out = ref.round_to_e4m3(np.array([value, -value], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([expected, -expected], dtype=np.float32))


@pytest.mark.parametrize(
    "value, bits",
    [
        (448.0, 0x7E),   # 0 1111 110: (1 + 6/8) * 2**(15-7)
        (1.0, 0x38),     # 0 0111 000
        (0.875, 0x36),   # 0.875 = 1.75 * 2**-1 -> 0 0110 110
        (-2.0, 0xC0),    # 1 1000 000
        (2.0 ** -9, 0x01),  # smallest subnormal: 0 0000 001
    ],
)
def test_e4m3_bit_encoding_hand_examples(value, bits):
    assert ref.encode_e4m3(np.float32(value)) == bits
    assert ref.decode_e4m3(np.uint8(bits)) == np.float32(value)


def test_e4m3_all_codes_round_trip():
    # Every finite 8-bit pattern (0x7F and 0xFF are NaN) decodes to a value that
    # is already on the E4M3 grid and re-encodes to the same bits.
    bits = np.array([b for b in range(256) if b not in (0x7F, 0xFF)], dtype=np.uint8)
    values = ref.decode_e4m3(bits)
    np.testing.assert_array_equal(ref.round_to_e4m3(values), values)
    np.testing.assert_array_equal(ref.encode_e4m3(values), bits)


# ----------------------------------------------------------------------------
# FP4 E2M1 element rounding
# ----------------------------------------------------------------------------

def test_e2m1_grid_values_encode_exactly():
    grid = ref.E2M1_GRID
    np.testing.assert_array_equal(ref.e2m1_encode(grid), np.arange(8))
    # Negative values set the sign bit (bit 3); -0 is encoded as +0.
    np.testing.assert_array_equal(ref.e2m1_encode(-grid), [0, 9, 10, 11, 12, 13, 14, 15])
    np.testing.assert_array_equal(ref.e2m1_decode(np.arange(16)), ref.E2M1_VALUES)


def test_e2m1_ties_round_to_even():
    # Midpoints between neighbours; the tie goes to the code with mantissa bit 0.
    midpoints = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
    expected = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], dtype=np.float32)
    np.testing.assert_array_equal(ref.e2m1_decode(ref.e2m1_encode(midpoints)), expected)
    np.testing.assert_array_equal(ref.e2m1_decode(ref.e2m1_encode(-midpoints)), -expected)


def test_e2m1_non_ties_round_to_nearest():
    v = np.array([0.26, 0.74, 1.3, 2.4, 2.6, 4.9, 5.1], dtype=np.float32)
    expected = np.array([0.5, 0.5, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    np.testing.assert_array_equal(ref.e2m1_decode(ref.e2m1_encode(v)), expected)


def test_e2m1_clamps_and_signed_zero():
    v = np.array([7.0, 100.0, -6.5, -0.1, -0.0], dtype=np.float32)
    codes = ref.e2m1_encode(v)
    np.testing.assert_array_equal(codes, [7, 7, 15, 0, 0])  # -0.1 rounds to +0


# ----------------------------------------------------------------------------
# Packing
# ----------------------------------------------------------------------------

def test_pack_hand_example():
    # Element 2i -> low nibble, element 2i+1 -> high nibble.
    np.testing.assert_array_equal(ref.pack_fp4(np.array([1, 2, 3, 4])), [0x21, 0x43])
    # Odd length gets a zero code appended.
    np.testing.assert_array_equal(ref.pack_fp4(np.array([1, 2, 0xF])), [0x21, 0x0F])


def test_pack_unpack_round_trip():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=(5, 37), dtype=np.uint8)
    packed = ref.pack_fp4(codes)
    assert packed.shape == (5, 19)
    np.testing.assert_array_equal(ref.unpack_fp4(packed, 37), codes)


# ----------------------------------------------------------------------------
# NVFP4 quantize / dequantize: hand-computed example
# ----------------------------------------------------------------------------

def _hand_example():
    """A (1, 32) tensor = two blocks of 16 along axis 1.

    Block 0 is every E2M1 value times 0.875, so amax = 6 * 0.875 = 5.25 and
      tensor_scale = 5.25 / 2688 = 2**-9                    (exact)
      block_scale  = (5.25 / 6) / 2**-9 = 448                (E4M3 max, exact)
      total scale  = 448 * 2**-9 = 0.875 -> codes are exactly 0..15.

    Block 1 = [0.7, 0.3, 0.1, -0.05, 0, ...]:
      block_scale  = (0.7 / 6) / 2**-9 = 59.73 -> E4M3 (spacing 4 in [32, 64)) -> 60
      total scale  = 60 * 2**-9 = 0.1171875
       0.7  / s =  5.97 -> 6   (code 7)  -> 0.703125
       0.3  / s =  2.56 -> 3   (code 5)  -> 0.3515625
       0.1  / s =  0.85 -> 1   (code 2)  -> 0.1171875
      -0.05 / s = -0.43 -> -0.5 (code 9) -> -0.05859375
    """
    block0 = 0.875 * np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    block1 = np.zeros(16)
    block1[:4] = [0.7, 0.3, 0.1, -0.05]
    x = np.concatenate([block0, block1]).astype(np.float32)[None, :]
    expected_codes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15,
                               7, 5, 2, 9] + [0] * 12, dtype=np.uint8)
    expected_deq = np.concatenate(
        [block0, [0.703125, 0.3515625, 0.1171875, -0.05859375], np.zeros(12)]
    ).astype(np.float32)[None, :]
    return x, expected_codes, expected_deq


def test_quantize_hand_example():
    x, expected_codes, expected_deq = _hand_example()
    q = ref.quantize_nvfp4(x, axis=1)

    assert q.tensor_scale == np.float32(2.0 ** -9)
    np.testing.assert_array_equal(q.block_scales, [[448.0, 60.0]])
    np.testing.assert_array_equal(ref.unpack_fp4(q.packed), expected_codes[None, :])
    # First packed bytes: (0,1)->0x10, (2,3)->0x32, ..., (14,15)->0xFE
    np.testing.assert_array_equal(
        q.packed[0, :8], [0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE]
    )
    np.testing.assert_array_equal(ref.dequantize(q), expected_deq)
    # 32 elements * 4 bits = 16 bytes, + 2 one-byte block scales + 4-byte tensor scale
    assert q.nbytes == 16 + 2 + 4


def test_quantize_along_axis_0_matches_transpose():
    x, _, expected_deq = _hand_example()
    np.testing.assert_array_equal(ref.fake_quant(x.T, axis=0), expected_deq.T)

    rng = np.random.default_rng(1)
    y = rng.standard_normal((40, 24)).astype(np.float32)
    np.testing.assert_array_equal(ref.fake_quant(y, axis=0), ref.fake_quant(y.T, axis=1).T)
    np.testing.assert_array_equal(ref.fake_quant(y, axis=-1), ref.fake_quant(y, axis=1))


# ----------------------------------------------------------------------------
# NVFP4: padding, round trips, error bounds, edge cases
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("shape, axis", [((3, 37), 1), ((37, 3), 0), ((2, 5, 33), 1), ((1, 1), 0)])
def test_padding_odd_sizes(shape, axis):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(shape).astype(np.float32)
    q = ref.quantize_nvfp4(x, axis=axis)
    n = shape[axis]
    num_blocks = -(-n // 16)
    assert q.block_scales.shape[-1] == num_blocks
    assert q.packed.shape[-1] == num_blocks * 16 // 2

    # Padding with zeros explicitly must give the same answer (zeros never
    # change any amax), so quantizing the padded tensor and slicing matches.
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, num_blocks * 16 - n)
    padded = np.pad(x, pad)
    expected = np.take(ref.fake_quant(padded, axis=axis), np.arange(n), axis=axis)
    out = ref.fake_quant(x, axis=axis)
    assert out.shape == x.shape
    np.testing.assert_array_equal(out, expected)


def test_round_trip_of_representable_tensor_is_exact():
    """Build a tensor that NVFP4 can represent exactly and check that
    quantize(dequantize(.)) reproduces the same codes and scales."""
    rng = np.random.default_rng(3)
    rows, blocks = 4, 3
    codes = rng.integers(0, 16, size=(rows, blocks, 16), dtype=np.uint8)
    codes[codes == 8] = 0                    # we never produce -0
    codes[..., 0] = 7                        # each block holds a +6 so its amax is 6 * scale
    e4m3_values = ref.decode_e4m3(np.arange(0x08, 0x7F, dtype=np.uint8))  # normal positives
    block_scales = rng.choice(e4m3_values, size=(rows, blocks)).astype(np.float32)
    block_scales[0, 0] = 448.0               # the largest block scale must be 448
    tensor_scale = np.float32(2.0 ** -7)

    x = (ref.e2m1_decode(codes) * (block_scales * tensor_scale)[..., None]).reshape(rows, -1)
    q = ref.quantize_nvfp4(x, axis=1)
    assert q.tensor_scale == tensor_scale
    np.testing.assert_array_equal(q.block_scales, block_scales)
    np.testing.assert_array_equal(ref.unpack_fp4(q.packed).reshape(codes.shape), codes)
    np.testing.assert_array_equal(ref.dequantize(q), x)


def test_error_is_bounded_by_block_scale():
    """Each dequantized element is within one block scale of the input.

    The largest gap on the E2M1 grid is 2 (between 4 and 6), so rounding error is
    at most 1 * scale; clamping after E4M3 scale rounding adds at most
    6 * 2**-4 = 0.375 * scale, so 1 * scale bounds both."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 100)).astype(np.float32)
    x[0, 3] = 40.0  # an outlier
    q = ref.quantize_nvfp4(x, axis=1)
    scale = np.repeat(q.block_scales * q.tensor_scale, 16, axis=1)[:, :100]
    err = np.abs(ref.dequantize(q) - x)
    assert np.all(err <= scale * (1 + 1e-6))
    # Sanity: the largest block scale is exactly the E4M3 max.
    assert q.block_scales.max() == 448.0


def test_all_zero_tensor():
    x = np.zeros((2, 20), dtype=np.float32)
    q = ref.quantize_nvfp4(x, axis=1)
    assert q.tensor_scale == 1.0
    assert not np.any(q.packed)
    np.testing.assert_array_equal(ref.dequantize(q), x)


def test_tiny_block_underflows_to_zero_without_nan():
    # One huge value sets the tensor scale; block 1's scale relative to it is
    # (1e-3/6) / (1e6/2688) ~ 4.5e-4 < 2**-10, so it rounds to 0 in E4M3.
    x = np.full((1, 32), 1e-3, dtype=np.float32)
    x[0, 0] = 1e6
    q = ref.quantize_nvfp4(x, axis=1)
    assert q.block_scales[0, 1] == 0.0
    out = ref.dequantize(q)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[0, 16:], 0.0)


# ----------------------------------------------------------------------------
# Stochastic rounding
# ----------------------------------------------------------------------------

def test_sr_leaves_representable_values_unchanged():
    x, _, _ = _hand_example()
    block0 = x[:, :16]   # exactly representable (see _hand_example)
    for seed in range(5):
        out = ref.fake_quant(block0, axis=1, stochastic=True, rng=np.random.default_rng(seed))
        np.testing.assert_array_equal(out, block0)


def test_sr_only_picks_neighbouring_values():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 48)).astype(np.float32)
    # rand = 0 always rounds up (when not on the grid); rand ~ 1 always rounds down.
    ceil = ref.fake_quant(x, axis=1, stochastic=True, rand=np.zeros_like(x))
    floor = ref.fake_quant(x, axis=1, stochastic=True,
                           rand=np.full_like(x, np.float32(1 - 2 ** -24)))
    assert np.all(np.abs(floor) <= np.abs(x) + 1e-6)
    for seed in range(10):
        out = ref.fake_quant(x, axis=1, stochastic=True, rng=np.random.default_rng(seed))
        assert np.all((out == ceil) | (out == floor))


def test_sr_is_reproducible_with_same_seed():
    x = np.random.default_rng(6).standard_normal((3, 40)).astype(np.float32)
    a = ref.fake_quant(x, axis=1, stochastic=True, rng=np.random.default_rng(42))
    b = ref.fake_quant(x, axis=1, stochastic=True, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_sr_is_unbiased_on_average():
    """E[SR(x)] should equal x (up to clamping at +-6 * scale). Round-to-nearest
    is biased; SR's average error should be far smaller."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    q = ref.quantize_nvfp4(x, axis=1)            # scales don't depend on rounding mode
    scale = np.repeat(q.block_scales * q.tensor_scale, 16, axis=1)
    expected = np.clip(x / scale, -6, 6) * scale  # what an unbiased rounder averages to

    trials = 2000
    mean = np.zeros_like(x, dtype=np.float64)
    for _ in range(trials):
        mean += ref.fake_quant(x, axis=1, stochastic=True, rng=rng)
    mean /= trials

    # Per-element std is at most (largest gap / 2) * scale = scale, so allow 5 sigma.
    assert np.all(np.abs(mean - expected) <= 5 * scale / np.sqrt(trials))

    sr_bias = np.mean(np.abs(mean - expected))
    rn_bias = np.mean(np.abs(ref.fake_quant(x, axis=1) - expected))
    assert sr_bias < 0.2 * rn_bias
