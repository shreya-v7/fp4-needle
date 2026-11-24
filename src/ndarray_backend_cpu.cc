#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "nvfp4.h"  // NVFP4 arithmetic shared with the CUDA backend

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();

  for (size_t i = 0; i < out->size; i++) {
      size_t index = i;
      size_t position = offset;

      for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
          size_t coord = index % shape[d];
          index /= shape[d];

          position += coord * strides[d];
      }

      out->ptr[i] = a.ptr[position];
  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();

  for (size_t i = 0; i < a.size; i++) {
      size_t index = i;
      size_t position = offset;

      for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
          size_t coord = index % shape[d];
          index /= shape[d];

          position += coord * strides[d];
      }

      out->ptr[position] = a.ptr[i];
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  size_t ndim = shape.size();

  for (size_t i = 0; i < size; i++) {
      size_t index = i;
      size_t position = offset;

      for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
          size_t coord = index % shape[d];
          index /= shape[d];

          position += coord * strides[d];
      }

      out->ptr[position] = val;
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] * b.ptr[i];
    }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] * val;
    }
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] / b.ptr[i];
    }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] / val;
    }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::pow(a.ptr[i], val);
    }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
    }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::max(a.ptr[i], val);
    }
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (a.ptr[i] == b.ptr[i]) ? 1.0f : 0.0f;
    }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (a.ptr[i] == val) ? 1.0f : 0.0f;
    }
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (a.ptr[i] >= b.ptr[i]) ? 1.0f : 0.0f;
    }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (a.ptr[i] >= val) ? 1.0f : 0.0f;
    }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::log(a.ptr[i]);
    }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::exp(a.ptr[i]);
    }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::tanh(a.ptr[i]);
    }
}

// Added for FP4 quantization: floor and absolute value, used to round
// block scales to FP8 E4M3 (see python/needle/fp4/quantize.py).
void EwiseFloor(const AlignedArray& a, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::floor(a.ptr[i]);
    }
}

void EwiseAbs(const AlignedArray& a, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::fabs(a.ptr[i]);
    }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
        scalar_t sum = 0.0f;

          for (size_t k = 0; k < n; k++) {
              sum += a.ptr[i * n + k] * b.ptr[k * p + j];
          }

          out->ptr[i * p + j] = sum;
      }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (size_t i = 0; i < TILE; i++) {
    for (size_t k = 0; k < TILE; k++) {
        scalar_t a_val = a[i * TILE + k];

          for (size_t j = 0; j < TILE; j++) {
              out[i * TILE + j] += a_val * b[k * TILE + j];
          }
      }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  size_t mt = m / TILE;
  size_t nt = n / TILE;
  size_t pt = p / TILE;

  for (size_t i = 0; i < out->size; i++) {
      out->ptr[i] = 0.0f;
  }

  for (size_t i = 0; i < mt; i++) {
      for (size_t j = 0; j < pt; j++) {
          scalar_t* out_tile = out->ptr + (i * pt + j) * TILE * TILE;

          for (size_t k = 0; k < nt; k++) {
              const scalar_t* a_tile =
                  a.ptr + (i * nt + k) * TILE * TILE;

              const scalar_t* b_tile =
                  b.ptr + (k * pt + j) * TILE * TILE;

              AlignedDot(a_tile, b_tile, out_tile);
          }
      }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    scalar_t max_val = a.ptr[i * reduce_size];

      for (size_t j = 1; j < reduce_size; j++) {
          max_val = std::max(max_val, a.ptr[i * reduce_size + j]);
      }

      out->ptr[i] = max_val;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    scalar_t sum = 0.0f;

      for (size_t j = 0; j < reduce_size; j++) {
          sum += a.ptr[i * reduce_size + j];
      }

      out->ptr[i] = sum;
  }
  /// END SOLUTION
}


////////////////////////////////////////////////////////////////////////////////
// NVFP4 (Stage 4): packed 4-bit storage, quantize+pack, dequantize, GEMV
//
// A matrix W of shape (rows, cols) is stored blocked along `cols` (each row's
// reduction axis for y = W x):
//   packed: rows * padded_cols / 2 bytes, two 4-bit codes per byte; element 2i
//           in the low nibble, element 2i+1 in the high nibble
//   scales: rows * padded_cols / 16 bytes, one E4M3 block scale per 16 elements
//   tensor_scale: one float
// Columns are zero-padded up to a multiple of 16 (nvfp4::padded_cols).
//
// These CPU versions are the readable, step-by-step versions of the CUDA
// kernels in ndarray_backend_cuda.cu, and produce bit-identical results.
////////////////////////////////////////////////////////////////////////////////

/** Plain byte buffer for packed codes and E4M3 scales (Array only holds floats). */
struct ByteArray {
  ByteArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size == 0 ? 1 : size);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~ByteArray() { free(ptr); }
  uint8_t* ptr;
  size_t size;
};

float Nvfp4QuantizePack(const AlignedArray& a, size_t rows, size_t cols,
                        const AlignedArray* rand, ByteArray* packed, ByteArray* scales) {
  /**
   * Quantize a compact (rows, cols) float matrix to NVFP4 and pack it.
   *
   * Args:
   *   a: input matrix, row-major
   *   rand: uniforms in [0, 1) with a's shape for stochastic rounding, or None
   *   packed, scales: outputs, sized as described above
   * Returns:
   *   the per-tensor FP32 scale
   */
  using namespace nvfp4;
  size_t cols_pad = padded_cols(cols);

  // 1. Per-tensor scale from the absolute maximum of the whole matrix.
  float amax = 0.0f;
  for (size_t i = 0; i < rows * cols; i++) amax = std::fmax(amax, std::fabs(a.ptr[i]));
  float ts = tensor_scale_from_amax(amax);

  for (size_t r = 0; r < rows; r++) {
    for (size_t c0 = 0; c0 < cols_pad; c0 += BLOCK) {
      // 2. Block amax over 16 elements (padding counts as 0).
      float block[BLOCK], block_amax = 0.0f;
      for (int j = 0; j < BLOCK; j++) {
        size_t c = c0 + j;
        block[j] = c < cols ? a.ptr[r * cols + c] : 0.0f;
        block_amax = std::fmax(block_amax, std::fabs(block[j]));
      }
      // 3. E4M3 block scale; the element scale is block scale * tensor scale.
      float bs = block_scale(block_amax, ts);
      float scale = mul_rn(bs, ts);
      size_t block_id = (r * cols_pad + c0) / BLOCK;
      scales->ptr[block_id] = encode_e4m3(bs);

      // 4. Round each element to a 4-bit code and pack pairs into bytes.
      uint8_t codes[BLOCK];
      for (int j = 0; j < BLOCK; j++) {
        size_t c = c0 + j;
        float v = scale > 0.0f ? div_rn(block[j], scale) : 0.0f;
        float u = (rand != nullptr && c < cols) ? rand->ptr[r * cols + c] : 0.0f;
        codes[j] = e2m1_encode(v, rand != nullptr, u);
      }
      for (int j = 0; j < BLOCK; j += 2)
        packed->ptr[(r * cols_pad + c0 + j) / 2] = codes[j] | (codes[j + 1] << 4);
    }
  }
  return ts;
}

void Nvfp4Dequantize(const ByteArray& packed, const ByteArray& scales, float ts, size_t rows,
                     size_t cols, AlignedArray* out) {
  /** Expand a packed NVFP4 matrix back to a compact (rows, cols) float matrix. */
  using namespace nvfp4;
  size_t cols_pad = padded_cols(cols);
  for (size_t r = 0; r < rows; r++) {
    for (size_t c = 0; c < cols; c++) {
      size_t i = r * cols_pad + c;                       // index in the padded matrix
      uint8_t byte = packed.ptr[i / 2];
      int code = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
      float scale = mul_rn(decode_e4m3(scales.ptr[i / BLOCK]), ts);
      out->ptr[r * cols + c] = mul_rn(e2m1_value(code), scale);
    }
  }
}

void Nvfp4Gemv(const ByteArray& packed, const ByteArray& scales, float ts, const AlignedArray& x,
               AlignedArray* y, size_t rows, size_t cols) {
  /**
   * y = W x with W in packed NVFP4, x and y in float32.
   *
   * The summation order is fixed so the CPU, the CUDA kernel and the NumPy
   * reference (reference.nvfp4_gemv) agree bit for bit. It is the order of the
   * CUDA kernel, where one warp of 32 "lanes" computes one output row:
   *   - lane l handles blocks l, l+32, l+64, ... of the row;
   *   - per block: s = sum over its 16 elements of table[code] * x, in order,
   *     then lane_acc += s * (block scale * tensor scale);
   *   - finally the 32 lane sums are combined by a butterfly: for offset
   *     16, 8, 4, 2, 1, every lane adds the value of lane (l XOR offset).
   */
  using namespace nvfp4;
  const int LANES = 32;
  size_t cols_pad = padded_cols(cols), num_blocks = cols_pad / BLOCK;
  for (size_t r = 0; r < rows; r++) {
    float lane_acc[LANES] = {0.0f};
    for (size_t b = 0; b < num_blocks; b++) {
      float scale = mul_rn(decode_e4m3(scales.ptr[r * num_blocks + b]), ts);
      float s = 0.0f;
      for (int j = 0; j < BLOCK; j++) {
        size_t c = b * BLOCK + j, i = r * cols_pad + c;
        uint8_t byte = packed.ptr[i / 2];
        int code = (j % 2 == 0) ? (byte & 0xF) : (byte >> 4);
        float xv = c < cols ? x.ptr[c] : 0.0f;
        s = add_rn(s, mul_rn(e2m1_value(code), xv));
      }
      int lane = b % LANES;
      lane_acc[lane] = add_rn(lane_acc[lane], mul_rn(s, scale));
    }
    for (int offset = LANES / 2; offset > 0; offset /= 2) {  // butterfly reduction
      float next[LANES];
      for (int l = 0; l < LANES; l++) next[l] = add_rn(lane_acc[l], lane_acc[l ^ offset]);
      for (int l = 0; l < LANES; l++) lane_acc[l] = next[l];
    }
    y->ptr[r] = lane_acc[0];
  }
}

void GemvFp32(const AlignedArray& w, const AlignedArray& x, AlignedArray* y, size_t rows,
              size_t cols) {
  /** Plain float32 y = W x (the baseline the FP4 GEMV is compared with). */
  for (size_t r = 0; r < rows; r++) {
    float acc = 0.0f;
    for (size_t c = 0; c < cols; c++) acc += w.ptr[r * cols + c] * x.ptr[c];
    y->ptr[r] = acc;
  }
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);
  m.def("ewise_floor", EwiseFloor);
  m.def("ewise_abs", EwiseAbs);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  // NVFP4 (Stage 4)
  py::class_<ByteArray>(m, "ByteArray")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &ByteArray::size);
  m.def("bytes_to_numpy", [](const ByteArray& a) {
    return py::array_t<uint8_t>(a.size, a.ptr);   // copies
  });
  m.def("bytes_from_numpy", [](py::array_t<uint8_t> a, ByteArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size);
  });
  m.def("nvfp4_quantize_pack", Nvfp4QuantizePack, py::arg("a"), py::arg("rows"),
        py::arg("cols"), py::arg("rand").none(true), py::arg("packed"), py::arg("scales"));
  m.def("nvfp4_dequantize", Nvfp4Dequantize);
  m.def("nvfp4_gemv", Nvfp4Gemv);
  m.def("gemv_fp32", GemvFp32);
}
