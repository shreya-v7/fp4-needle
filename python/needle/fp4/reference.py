# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""NumPy reference implementation of NVFP4 quantization.

This file is the "ground truth" for the project: the needle fake-quant Op
(Stage 2) and the CUDA kernels (Stage 4) are tested against it. It is written
for clarity, not speed, and uses only NumPy.

Format summary (NVFP4)
----------------------
* Each element is FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit.
  The 8 representable magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}; there is no
  inf/NaN and the maximum is 6.
* Elements are grouped into blocks of 16 contiguous values along the matmul's
  *reduction* axis. Each block has its own scale stored in FP8 E4M3.
* The whole tensor also has one FP32 scale, chosen so that the largest block
  scale lands exactly on E4M3's maximum (448):
        tensor_scale = amax(tensor) / (6 * 448)
        block_scale  = round_to_e4m3( (amax(block) / 6) / tensor_scale )
        element      ~= e2m1_value(code) * block_scale * tensor_scale

Rounding conventions (decided in Stage 0)
-----------------------------------------
* Deterministic rounding is round-to-nearest, ties-to-even (the tie goes to
  the code whose mantissa bit is 0). This matches NVIDIA's cvt.rn.e2m1.
* Stochastic rounding (SR) applies only to the FP4 elements: a magnitude
  between two grid points lo <= a <= hi rounds up with probability
  (a - lo) / (hi - lo). Block scales are always round-to-nearest-even.
* SR randomness is passed in as an array of uniforms `rand` (same shape as x)
  so that other implementations can be checked against this one bit-for-bit.
* Zero is always encoded as +0 (code 0), even for negative inputs that round
  to zero, so every implementation produces identical bytes.
* All arithmetic is float32, in the order written here, so the CUDA kernel can
  reproduce it exactly.

Packing: two 4-bit codes per uint8 byte; element 2i goes in the low nibble and
element 2i+1 in the high nibble. Bit 3 of each nibble is the sign bit.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# Format constants
# ----------------------------------------------------------------------------

#: Representable E2M1 magnitudes, indexed by the 3 low bits of a code
#: (exponent bits 2..1, mantissa bit 0). Index 1 is the only subnormal (0.5).
E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

#: Value of each of the 16 FP4 codes (bit 3 = sign). Code 8 is "-0", which we
#: never produce but decode as 0.0. This is the lookup table the GEMV kernel uses.
E2M1_VALUES = np.concatenate([E2M1_GRID, -E2M1_GRID]).astype(np.float32)

FP4_MAX = np.float32(6.0)      # largest E2M1 magnitude
E4M3_MAX = np.float32(448.0)   # largest finite FP8 E4M3 value (S.1111.110)
E4M3_MIN_NORMAL_EXP = -6       # smallest normal exponent of E4M3 (bias 7)
E4M3_MANTISSA_BITS = 3
NVFP4_BLOCK_SIZE = 16


# ----------------------------------------------------------------------------
# FP8 E4M3 (used for the per-block scales)
# ----------------------------------------------------------------------------

def round_to_e4m3(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest FP8 E4M3 value (ties to even).

    Values above 448 saturate to 448 (E4M3 has no inf). Values below the
    smallest normal (2**-6) use E4M3's subnormal grid, whose spacing is 2**-9.
    Returns float32 values that are exactly representable in E4M3.
    """
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x)

    # frexp gives a = m * 2**e2 with m in [0.5, 1), so floor(log2(a)) = e2 - 1.
    # (Exact, unlike floor(np.log2(a)), which can be off by one near powers of 2.)
    _, e2 = np.frexp(a)
    exponent = np.maximum(e2 - 1, E4M3_MIN_NORMAL_EXP)

    # With 3 mantissa bits there are 8 steps per power of two, so the spacing
    # between neighbouring E4M3 values in this binade is 2**(exponent - 3).
    step = np.ldexp(np.float32(1.0), exponent - E4M3_MANTISSA_BITS).astype(np.float32)

    # a / step is exact (step is a power of two); np.round is half-to-even.
    rounded = (np.round(a / step) * step).astype(np.float32)
    rounded = np.minimum(rounded, E4M3_MAX)
    return np.copysign(rounded, x).astype(np.float32)


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Convert E4M3-representable float32 values to their 8-bit encoding.

    Bit layout: [sign:1][exponent:4][mantissa:3], exponent bias 7.
    Input values must already be on the E4M3 grid (use round_to_e4m3 first).
    """
    x = np.asarray(x, dtype=np.float32)
    assert np.array_equal(round_to_e4m3(x), x), "values must be E4M3-representable"
    sign = (np.signbit(x)).astype(np.uint8)
    a = np.abs(x)

    _, e2 = np.frexp(a)
    exponent = e2 - 1
    is_normal = a >= np.float32(2.0 ** E4M3_MIN_NORMAL_EXP)

    # Normal:    a = (1 + mant/8) * 2**exponent, exponent field = exponent + 7
    # Subnormal: a = (mant/8) * 2**-6,           exponent field = 0
    exp_field = np.where(is_normal, exponent + 7, 0)
    mant = np.where(
        is_normal,
        (np.ldexp(a, -exponent) - 1.0) * 8.0,
        np.ldexp(a, -E4M3_MIN_NORMAL_EXP) * 8.0,
    )
    return ((sign << 7) | (exp_field.astype(np.uint8) << 3) | mant.astype(np.uint8)).astype(np.uint8)


def decode_e4m3(bits: np.ndarray) -> np.ndarray:
    """Convert 8-bit E4M3 encodings back to float32 values (inverse of encode_e4m3)."""
    bits = np.asarray(bits, dtype=np.uint8)
    sign = np.where(bits >> 7, -1.0, 1.0).astype(np.float32)
    exp_field = ((bits >> 3) & 0xF).astype(np.int32)
    mant = (bits & 0x7).astype(np.float32)
    assert not np.any((exp_field == 15) & (mant == 7)), "0x7F / 0xFF are NaN in E4M3"

    normal = np.ldexp(1.0 + mant / 8.0, exp_field - 7)
    subnormal = np.ldexp(mant / 8.0, E4M3_MIN_NORMAL_EXP)
    return (sign * np.where(exp_field > 0, normal, subnormal)).astype(np.float32)


# ----------------------------------------------------------------------------
# FP4 E2M1 (the elements)
# ----------------------------------------------------------------------------

def e2m1_encode(v: np.ndarray, stochastic: bool = False,
                rand: Optional[np.ndarray] = None) -> np.ndarray:
    """Round already-scaled float32 values to 4-bit E2M1 codes (uint8 in 0..15).

    Args:
        v: values in "FP4 units" (i.e. already divided by the scale).
           They are clamped to [-6, 6] first.
        stochastic: use stochastic rounding instead of round-to-nearest-even.
        rand: uniforms in [0, 1), same shape as v (required if stochastic).
    """
    v = np.clip(np.asarray(v, dtype=np.float32), -FP4_MAX, FP4_MAX)
    a = np.abs(v)

    # Bracket |v| between two neighbouring grid points: lo <= a <= hi.
    lo = np.searchsorted(E2M1_GRID, a, side="right") - 1   # largest grid value <= a
    hi = np.minimum(lo + 1, len(E2M1_GRID) - 1)            # next one up (6 stays 6)
    lo_val, hi_val = E2M1_GRID[lo], E2M1_GRID[hi]

    if stochastic:
        assert rand is not None and rand.shape == v.shape
        gap = hi_val - lo_val
        # Probability of rounding up = fractional distance to the next value.
        # When a == 6 there is no next value (gap == 0) and we never round up.
        p_up = np.where(gap > 0, (a - lo_val) / np.where(gap > 0, gap, 1), 0)
        round_up = rand < p_up
    else:
        mid = (lo_val + hi_val) / np.float32(2)
        # Ties go to the even index (mantissa bit 0), e.g. 2.5 -> 2 and 5 -> 4.
        round_up = (a > mid) | ((a == mid) & (hi % 2 == 0))

    magnitude_idx = np.where(round_up, hi, lo).astype(np.uint8)
    # Set the sign bit only for nonzero results, so zero is always +0.
    sign_bit = ((v < 0) & (magnitude_idx > 0)).astype(np.uint8)
    return (sign_bit << 3) | magnitude_idx


def e2m1_decode(codes: np.ndarray) -> np.ndarray:
    """Look up the float32 value of each 4-bit code."""
    return E2M1_VALUES[np.asarray(codes, dtype=np.uint8)]


# ----------------------------------------------------------------------------
# Packing two 4-bit codes per byte
# ----------------------------------------------------------------------------

def pack_fp4(codes: np.ndarray) -> np.ndarray:
    """Pack 4-bit codes along the last axis: byte i = codes[2i] | codes[2i+1] << 4.

    If the last axis has odd length it is padded with one zero code.
    """
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] % 2:
        codes = np.concatenate([codes, np.zeros(codes.shape[:-1] + (1,), np.uint8)], axis=-1)
    low, high = codes[..., 0::2], codes[..., 1::2]
    return (low | (high << 4)).astype(np.uint8)


def unpack_fp4(packed: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """Inverse of pack_fp4. Returns codes along the last axis, truncated to n."""
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty(packed.shape[:-1] + (2 * packed.shape[-1],), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    return codes if n is None else codes[..., :n]


# ----------------------------------------------------------------------------
# NVFP4 quantize / dequantize
# ----------------------------------------------------------------------------

@dataclass
class NVFP4Tensor:
    """A tensor stored in NVFP4.

    The quantized (reduction) axis is moved to the last position internally and
    padded with zeros up to a multiple of the block size.

    Attributes:
        packed:       uint8, shape (*other_dims, padded_len // 2), two codes per byte.
        block_scales: float32, shape (*other_dims, num_blocks); every value is
                      exactly representable in E4M3 (see encode_e4m3 for bytes).
        tensor_scale: float32 scalar, the per-tensor FP32 scale.
        shape:        original shape of the tensor.
        axis:         the original axis that was blocked (non-negative).
    """
    packed: np.ndarray
    block_scales: np.ndarray
    tensor_scale: np.float32
    shape: tuple
    axis: int
    block_size: int = NVFP4_BLOCK_SIZE

    @property
    def nbytes(self) -> int:
        """Storage cost: 4 bits per element + 1 byte per block scale + 4-byte tensor scale."""
        return self.packed.nbytes + self.block_scales.size + 4


def _to_blocks(x: np.ndarray, axis: int, block_size: int) -> np.ndarray:
    """Move `axis` last, zero-pad it to a multiple of block_size, and split it
    into blocks: returns shape (*other_dims, num_blocks, block_size)."""
    x = np.moveaxis(x, axis, -1)
    pad = (-x.shape[-1]) % block_size
    x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    return x.reshape(x.shape[:-1] + (-1, block_size))


def compute_tensor_scale(x: np.ndarray) -> np.float32:
    """Per-tensor FP32 scale = amax / (6 * 448); 1.0 for an all-zero tensor."""
    amax = np.float32(np.max(np.abs(x))) if x.size else np.float32(0)
    return amax / (FP4_MAX * E4M3_MAX) if amax > 0 else np.float32(1.0)


def quantize_nvfp4(x: np.ndarray, axis: int = -1, stochastic: bool = False,
                   rand: Optional[np.ndarray] = None,
                   rng: Optional[np.random.Generator] = None) -> NVFP4Tensor:
    """Quantize x to NVFP4 with blocks of 16 along `axis`.

    Args:
        x: input array (converted to float32).
        axis: the axis to block along; this must be the matmul's reduction axis.
        stochastic: stochastic rounding of the FP4 elements.
        rand: optional uniforms in [0, 1) with x's shape (for reproducibility);
              if omitted and stochastic=True they are drawn from `rng`.
        rng: optional numpy Generator used when rand is None.
    """
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim >= 1, "quantize_nvfp4 needs at least a 1-D array"
    assert np.all(np.isfinite(x)), "NVFP4 has no inf/NaN; the input must be finite"
    axis = axis % x.ndim
    bs = NVFP4_BLOCK_SIZE

    # 1. Per-tensor FP32 scale.
    tensor_scale = compute_tensor_scale(x)

    # 2. Per-block scale: amax(block) / 6, relative to the tensor scale, in E4M3.
    blocks = _to_blocks(x, axis, bs)                 # (..., num_blocks, 16)
    block_amax = np.max(np.abs(blocks), axis=-1)     # (..., num_blocks)
    block_scales = round_to_e4m3((block_amax / FP4_MAX) / tensor_scale)

    # 3. Divide each element by its total scale. A block whose scale is 0
    #    (all zeros, or so tiny it underflows E4M3) encodes as all zeros.
    scale = (block_scales * tensor_scale)[..., None]  # (..., num_blocks, 1)
    has_scale = scale > 0
    scaled = np.where(has_scale, blocks / np.where(has_scale, scale, 1), 0).astype(np.float32)

    # 4. Round to E2M1 codes (clamping to +-6 happens inside e2m1_encode).
    rand_blocks = None
    if stochastic:
        if rand is None:
            rng = np.random.default_rng() if rng is None else rng
            rand = rng.random(x.shape, dtype=np.float32)
        rand_blocks = _to_blocks(np.asarray(rand, np.float32), axis, bs)
    codes = e2m1_encode(scaled, stochastic, rand_blocks)

    # 5. Pack along the (blocked) last axis.
    packed = pack_fp4(codes.reshape(codes.shape[:-2] + (-1,)))
    return NVFP4Tensor(packed, block_scales.astype(np.float32), np.float32(tensor_scale),
                       tuple(x.shape), axis, bs)


def dequantize(q: NVFP4Tensor) -> np.ndarray:
    """Reconstruct a float32 array with the original shape from an NVFP4Tensor."""
    codes = unpack_fp4(q.packed)                               # (..., padded_len)
    values = e2m1_decode(codes).reshape(codes.shape[:-1] + (-1, q.block_size))
    scale = (q.block_scales * q.tensor_scale)[..., None]       # same op order as quantize
    out = (values * scale).reshape(codes.shape).astype(np.float32)
    out = out[..., : q.shape[q.axis]]                          # drop the padding
    return np.moveaxis(out, -1, q.axis)


def fake_quant(x: np.ndarray, axis: int = -1, stochastic: bool = False,
               rand: Optional[np.ndarray] = None,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Quantize to NVFP4 and immediately dequantize ("fake quantization").

    The result is float32 with x's shape, but every value is one that NVFP4
    can represent. This is what training in FP4 "sees".
    """
    return dequantize(quantize_nvfp4(x, axis, stochastic, rand, rng))


# ----------------------------------------------------------------------------
# NVFP4 matrix-vector product (the reference for the Stage 4 GEMV kernels)
# ----------------------------------------------------------------------------

GEMV_LANES = 32   # one CUDA warp computes one output row


def nvfp4_gemv(q: NVFP4Tensor, x: np.ndarray) -> np.ndarray:
    """y = W x for W stored in NVFP4 (quantized along axis 1), x in float32.

    Floating-point addition is not associative, so to be bit-identical to the
    kernels this adds in exactly their order (see Nvfp4Gemv in the backends):
      1. per block of 16: s = sum of e2m1_value(code) * x, one element at a time;
      2. "lane" l of 32 accumulates blocks l, l+32, l+64, ...:
             lane_acc += s * (block_scale * tensor_scale);
      3. butterfly over lanes: for offset 16, 8, 4, 2, 1:
             lane_acc[l] = lane_acc[l] + lane_acc[l XOR offset].
    Each step is one float32 operation, rounded on its own.
    """
    assert len(q.shape) == 2 and q.axis == 1, "W must be 2-D and blocked along its columns"
    rows, cols = q.shape
    codes = unpack_fp4(q.packed)                                    # (rows, cols_pad)
    num_blocks = codes.shape[1] // q.block_size
    values = e2m1_decode(codes).reshape(rows, num_blocks, q.block_size)
    x_pad = np.zeros(num_blocks * q.block_size, dtype=np.float32)
    x_pad[:cols] = x
    x_blocks = x_pad.reshape(num_blocks, q.block_size)

    # 1. Per-block partial sums, accumulated element by element.
    s = np.zeros((rows, num_blocks), dtype=np.float32)
    for j in range(q.block_size):
        s = s + values[:, :, j] * x_blocks[:, j]
    scale = (q.block_scales * q.tensor_scale).astype(np.float32)    # (rows, num_blocks)

    # 2. Lane accumulation: block b goes to lane b % 32, in increasing b.
    lane_acc = np.zeros((rows, GEMV_LANES), dtype=np.float32)
    for b in range(num_blocks):
        lane_acc[:, b % GEMV_LANES] = lane_acc[:, b % GEMV_LANES] + s[:, b] * scale[:, b]

    # 3. Butterfly reduction across lanes.
    lanes = np.arange(GEMV_LANES)
    offset = GEMV_LANES // 2
    while offset:
        lane_acc = lane_acc + lane_acc[:, lanes ^ offset]
        offset //= 2
    return lane_acc[:, 0].astype(np.float32)
