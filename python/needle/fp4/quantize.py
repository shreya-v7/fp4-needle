# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""NVFP4 fake quantization on needle NDArrays (runs on any needle device).

This is the needle version of `reference.fake_quant`. It uses only NDArray
operations, so the same code runs on the cpu, cuda, and cpu_numpy devices, and
it follows the reference step by step so the results are bit-identical
(tests/fp4/test_fake_quant.py checks this).

The only new backend primitives it needs are `floor` and `abs`. Everything
else is built from ops needle already had: comparisons (>=, ==), +, *, /,
maximum, reductions, permute/reshape/pad.

Why no log2/exp2? Rounding a block scale to E4M3 needs the power-of-two
"binade" of each value. floor(log2(x)) can be off by one near powers of two
and exp2 is not guaranteed exact on every GPU, either of which would break bit
equality with the reference. Instead we build the E4M3 step size as an exact
sum of powers of two using comparisons (see `round_to_e4m3`).
"""

from typing import Optional

from .reference import E4M3_MAX, FP4_MAX, NVFP4_BLOCK_SIZE

# E2M1 rounding table for round-to-nearest-even. Each row is
# (midpoint between two neighbouring grid values, size of the jump, tie rounds up?).
# A magnitude a rounds to the sum of the jumps whose midpoint it passes.
# Ties go to the even code: 0.75 -> 1, 1.75 -> 2, 3.5 -> 4 round up; the others
# (0.25 -> 0, 1.25 -> 1, 2.5 -> 2, 5 -> 4) round down.
_E2M1_RN_TABLE = [
    (0.25, 0.5, False),
    (0.75, 0.5, True),
    (1.25, 0.5, False),
    (1.75, 0.5, True),
    (2.5, 1.0, False),
    (3.5, 1.0, True),
    (5.0, 2.0, False),
]

# Grid points above zero and the jump from the previous grid point:
# lo(a) = sum of jumps for grid points <= a  is the largest grid value <= a.
_E2M1_FLOOR_TABLE = [(0.5, 0.5), (1.0, 0.5), (1.5, 0.5), (2.0, 0.5),
                     (3.0, 1.0), (4.0, 1.0), (6.0, 2.0)]


def _minimum(x, c: float):
    """Elementwise min(x, c) using the existing maximum: min(x, c) = -max(-x, -c)."""
    return -((-x).maximum(-c))


def _round_half_even(y):
    """Round nonnegative values to the nearest integer, ties to even (like np.round)."""
    whole = y.floor()
    frac = y - whole                                   # exact for y >= 0
    is_odd = whole - (whole / 2).floor() * 2           # 1.0 if whole is odd, else 0.0
    round_up = (frac > 0.5) + (frac == 0.5) * is_odd   # each term is 0.0 or 1.0
    return whole + round_up


def round_to_e4m3(a):
    """Round nonnegative values to the nearest FP8 E4M3 value (ties to even, max 448).

    Matches reference.round_to_e4m3 exactly. E4M3 has 3 mantissa bits, so in the
    binade [2**k, 2**(k+1)) neighbouring values are 2**(k-3) apart; below the
    smallest normal 2**-6 the (subnormal) spacing stays 2**-9. We build that
    spacing, `step`, without log2:
        step = 2**-9 + sum over k = -5..8 of [a >= 2**k] * 2**(k-4)
    For a in [2**k, 2**(k+1)) the sum telescopes to exactly 2**(k-3).
    (k stops at 8 because anything >= 512 saturates to 448 anyway.)
    """
    step = a.device.full(a.shape, 2.0 ** -9)
    for k in range(-5, 9):
        step = step + (a >= 2.0 ** k) * (2.0 ** (k - 4))
    rounded = _round_half_even(a / step) * step        # a / step is exact
    return _minimum(rounded, float(E4M3_MAX))


def round_to_e2m1(v, rand=None):
    """Round scaled values to the nearest E2M1 value; returns float values, not codes.

    Args:
        v: values already divided by their scale (any sign); clamped to [-6, 6].
        rand: if given, use stochastic rounding with these uniforms in [0, 1)
              (same shape as v); otherwise round to nearest, ties to even.
    """
    a = _minimum(v.abs(), float(FP4_MAX))

    if rand is None:
        # Deterministic: add up the jumps whose midpoint a has passed.
        q = a * 0.0
        for midpoint, jump, tie_rounds_up in _E2M1_RN_TABLE:
            passed = (a >= midpoint) if tie_rounds_up else (a > midpoint)
            q = q + passed * jump
    else:
        # Stochastic: lo = largest grid value <= a, gap = distance to the next one.
        lo = a * 0.0
        for grid_value, jump in _E2M1_FLOOR_TABLE:
            lo = lo + (a >= grid_value) * jump
        # gap is 0.5 below 2, 1 in [2, 4), 2 in [4, 6), and 0 at 6 (nothing above).
        gap = (lo >= 2.0) * 0.5 + (lo >= 4.0) * 1.0 - (lo >= 6.0) * 2.0 + 0.5
        # Round up with probability (a - lo) / gap. At a == 6 the numerator is 0,
        # so dividing by 1 instead of 0 is safe and gives probability 0.
        p_up = (a - lo) / (gap + (gap == 0.0))
        q = lo + gap * (rand < p_up)

    sign = (v >= 0.0) * 2.0 - 1.0                      # +1 or -1 (0 maps to +1)
    return q * sign


def _fake_quant_blocks(x, blocks, rand_blocks=None):
    """Steps 2 to 5 of fake_quant on blocks laid out along the last axis of `blocks`."""
    block_axis = len(blocks.shape) - 1

    # 2. Per-tensor FP32 scale = amax / (6 * 448), or 1 if the tensor is all zero.
    #    Kept as a 1-element NDArray (on the device) and broadcast where needed.
    tensor_amax = x.abs().max()                                   # shape (1,)
    tensor_scale = tensor_amax / float(FP4_MAX * E4M3_MAX) + (tensor_amax == 0.0)

    # 3. Per-block scale = round_to_e4m3((amax(block) / 6) / tensor_scale).
    block_amax = blocks.abs().max(axis=block_axis, keepdims=True)  # (..., nb, 1)
    tensor_scale = tensor_scale.reshape((1,) * len(block_amax.shape)).broadcast_to(block_amax.shape)
    block_scale = round_to_e4m3((block_amax / float(FP4_MAX)) / tensor_scale)

    # 4. Divide by the total scale. Blocks whose scale is 0 (all zero, or too
    #    small for E4M3) become zero: divide by 1 there, then multiply by 0.
    scale = block_scale * tensor_scale
    has_scale = (scale > 0.0).broadcast_to(blocks.shape)
    safe_scale = (scale + (scale == 0.0)).broadcast_to(blocks.shape)
    scaled = blocks / safe_scale * has_scale

    # 5. Round each element to E2M1 and multiply the scale back in.
    return round_to_e2m1(scaled, rand_blocks) * scale.broadcast_to(blocks.shape)


def fake_quant(x, axis: int, stochastic: bool = False, rand=None):
    """Quantize an NDArray to NVFP4 along `axis` and dequantize it again.

    Same contract as reference.fake_quant: blocks of 16 along `axis` (the
    matmul's reduction axis), zero-padded tails, per-tensor FP32 scale and
    per-block E4M3 scales. Returns a new compact NDArray with x's shape.

    Args:
        x: NDArray on any needle device.
        axis: axis to block along.
        stochastic: stochastic rounding of the FP4 elements.
        rand: optional NDArray of uniforms with x's shape; if stochastic and not
              given, drawn with x.device.rand (needle's usual random source).
    """
    bs = NVFP4_BLOCK_SIZE
    ndim = len(x.shape)
    axis = axis % ndim
    if stochastic and rand is None:
        rand = x.device.rand(*x.shape)
    if stochastic:
        assert tuple(rand.shape) == tuple(x.shape), f"rand has shape {rand.shape}, expected {x.shape}"

    # 1. Rearrange to (*other_dims, num_blocks, 16): move `axis` last, pad it
    #    with zeros to a multiple of 16, then split it into blocks.
    perm = tuple(i for i in range(ndim) if i != axis) + (axis,)
    inverse_perm = tuple(perm.index(i) for i in range(ndim))
    n = x.shape[axis]
    pad = (-n) % bs

    def to_blocks(t):
        t = t.permute(perm).compact()
        if pad:
            t = t.pad(((0, 0),) * (ndim - 1) + ((0, pad),))
        return t.reshape(t.shape[:-1] + ((n + pad) // bs, bs))

    blocks = to_blocks(x)
    rand_blocks = to_blocks(rand) if stochastic else None
    dequantized = _fake_quant_blocks(x, blocks, rand_blocks)

    # 6. Undo step 1: merge blocks, drop the padding, restore the axis order.
    out = dequantized.reshape(dequantized.shape[:-2] + (n + pad,))
    if pad:
        out = out[tuple(slice(None) for _ in range(ndim - 1)) + (slice(0, n),)]
    return out.compact().permute(inverse_perm).compact()


def fake_quant_2d(x):
    """NVFP4 fake quantization of a 2-D NDArray with one scale per 16 x 16 tile.

    Matches reference.fake_quant_2d bit for bit. Each scale covers a square tile,
    so the forward (rows) and backward (columns) matmuls see the same weight.
    """
    bs = NVFP4_BLOCK_SIZE
    rows, cols = x.shape
    pad_r, pad_c = (-rows) % bs, (-cols) % bs
    t = x.compact()
    if pad_r or pad_c:
        t = t.pad(((0, pad_r), (0, pad_c)))
    R, C = (rows + pad_r) // bs, (cols + pad_c) // bs
    tiles = t.reshape((R, bs, C, bs)).permute((0, 2, 1, 3)).compact().reshape((R, C, bs * bs))
    q = _fake_quant_blocks(x, tiles)
    out = q.reshape((R, C, bs, bs)).permute((0, 2, 1, 3)).compact().reshape((R * bs, C * bs))
    if pad_r or pad_c:
        out = out[0:rows, 0:cols]
    return out.compact()
