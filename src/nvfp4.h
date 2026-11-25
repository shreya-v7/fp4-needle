// AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
// NVFP4 arithmetic shared by the CPU backend (ndarray_backend_cpu.cc) and the
// CUDA backend (ndarray_backend_cuda.cu).
//
// Every function here mirrors python/needle/fp4/reference.py step by step, in
// float32 and in the same order, so the C++ and CUDA kernels can be tested for
// *bit-exact* agreement with the NumPy reference (tests/fp4/test_kernels.py).
//
// To keep the arithmetic identical on CPU and GPU:
//   * multiplies, adds and divides go through mul_rn / add_rn / div_rn. On the
//     GPU these are the "_rn" intrinsics, which round every operation on its
//     own and stop the compiler from fusing a*b+c into one FMA instruction
//     (which would round only once and change the result). On the CPU they are
//     plain operators; CMakeLists.txt passes -ffp-contract=off for the same reason.
//   * the E2M1 table is computed from the format definition instead of stored in
//     a global array, so the same code works in host and device functions.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define FP4_HD __host__ __device__ inline
#else
#define FP4_HD inline
#endif

namespace needle {
namespace nvfp4 {

const int BLOCK = 16;             // elements per block (along the reduction axis)
const float FP4_MAX = 6.0f;       // largest E2M1 magnitude
const float E4M3_MAX = 448.0f;    // largest finite FP8 E4M3 value

// ---------------------------------------------------------------------------
// Correctly rounded float32 operations (no FMA contraction on the GPU)
// ---------------------------------------------------------------------------
FP4_HD float mul_rn(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
FP4_HD float add_rn(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}
FP4_HD float div_rn(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}

// ---------------------------------------------------------------------------
// FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
// ---------------------------------------------------------------------------

// Magnitude of the 3-bit index idx = [exponent:2][mantissa:1]:
//   exponent 0 (subnormal): mantissa * 0.5                 -> 0, 0.5
//   exponent e > 0:         (1 + mantissa * 0.5) * 2^(e-1)  -> 1, 1.5, 2, 3, 4, 6
FP4_HD float e2m1_magnitude(int idx) {
  int e = idx >> 1, m = idx & 1;
  return e == 0 ? 0.5f * m : ldexpf(1.0f + 0.5f * m, e - 1);
}

// Value of a 4-bit code; bit 3 is the sign. This is the GEMV's 16-entry table.
FP4_HD float e2m1_value(int code) {
  float mag = e2m1_magnitude(code & 7);
  return (code & 8) ? -mag : mag;
}

// Round a scaled value to a 4-bit code. Same rules as reference.e2m1_encode:
// clamp to [-6, 6]; round to nearest with ties to the even code, or, if
// `stochastic`, round up with probability (a - lo) / (hi - lo) using `rand` in
// [0, 1); zero is always encoded as +0.
FP4_HD uint8_t e2m1_encode(float v, bool stochastic, float rand) {
  v = fminf(fmaxf(v, -FP4_MAX), FP4_MAX);
  float a = fabsf(v);
  int lo = 0;                                   // index of the largest grid value <= a
  for (int i = 1; i < 8; i++)
    if (a >= e2m1_magnitude(i)) lo = i;
  int hi = lo < 7 ? lo + 1 : 7;                 // next grid value up (6 stays 6)
  float lo_v = e2m1_magnitude(lo), hi_v = e2m1_magnitude(hi);
  bool up;
  if (stochastic) {
    float gap = hi_v - lo_v;
    float p_up = gap > 0.0f ? div_rn(a - lo_v, gap) : 0.0f;
    up = rand < p_up;
  } else {
    float mid = div_rn(lo_v + hi_v, 2.0f);
    up = a > mid || (a == mid && hi % 2 == 0);
  }
  int idx = up ? hi : lo;
  int sign = (v < 0.0f && idx > 0) ? 8 : 0;
  return (uint8_t)(sign | idx);
}

// ---------------------------------------------------------------------------
// FP8 E4M3 (the block scales): 1 sign, 4 exponent bits (bias 7), 3 mantissa bits
// ---------------------------------------------------------------------------

// Round a nonnegative value to the nearest E4M3 value (ties to even, max 448).
// Same steps as reference.round_to_e4m3.
FP4_HD float round_to_e4m3(float a) {
  if (!(a > 0.0f)) return 0.0f;
  int e2;
  frexpf(a, &e2);                               // a = m * 2^e2, m in [0.5, 1)
  int exponent = e2 - 1;                        // floor(log2(a)), exactly
  if (exponent < -6) exponent = -6;             // subnormal range keeps spacing 2^-9
  float step = ldexpf(1.0f, exponent - 3);      // 3 mantissa bits -> 8 steps per binade
  float q = mul_rn(rintf(div_rn(a, step)), step);   // rintf: round half to even
  return fminf(q, E4M3_MAX);
}

// 8-bit encoding of a nonnegative E4M3 value (inverse of decode_e4m3).
FP4_HD uint8_t encode_e4m3(float q) {
  int exp_field, mant;
  if (q < ldexpf(1.0f, -6)) {                   // subnormal: q = mant/8 * 2^-6
    exp_field = 0;
    mant = (int)div_rn(q, ldexpf(1.0f, -9));
  } else {                                      // normal: q = (1 + mant/8) * 2^(exp_field-7)
    int e2;
    frexpf(q, &e2);
    int e = e2 - 1;
    exp_field = e + 7;
    mant = (int)mul_rn(div_rn(q, ldexpf(1.0f, e)) - 1.0f, 8.0f);
  }
  return (uint8_t)((exp_field << 3) | mant);
}

FP4_HD float decode_e4m3(uint8_t bits) {
  int exp_field = (bits >> 3) & 0xF, mant = bits & 0x7;
  return exp_field ? ldexpf(1.0f + mant / 8.0f, exp_field - 7)
                   : ldexpf(mant / 8.0f, -6);
}

// ---------------------------------------------------------------------------
// NVFP4 scales
// ---------------------------------------------------------------------------

// Per-tensor FP32 scale = amax / (6 * 448), or 1 for an all-zero tensor.
FP4_HD float tensor_scale_from_amax(float amax) {
  return amax > 0.0f ? div_rn(amax, FP4_MAX * E4M3_MAX) : 1.0f;
}

// Per-block E4M3 scale = round_to_e4m3((amax(block) / 6) / tensor_scale).
FP4_HD float block_scale(float block_amax, float tensor_scale) {
  return round_to_e4m3(div_rn(div_rn(block_amax, FP4_MAX), tensor_scale));
}

// Round up to a whole number of blocks: the padded length of a row.
FP4_HD size_t padded_cols(size_t cols) { return (cols + BLOCK - 1) / BLOCK * BLOCK; }

}  // namespace nvfp4
}  // namespace needle
