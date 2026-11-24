#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

#include "nvfp4.h"  // NVFP4 arithmetic shared with the CPU backend

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ size_t GetIndex(
    size_t gid,
    CudaVec shape,
    CudaVec strides,
    size_t offset
) {
    size_t position = offset;

    for (int d = static_cast<int>(shape.size) - 1; d >= 0; d--) {
        size_t coord = gid % shape.data[d];
        gid /= shape.data[d];

        position += coord * strides.data[d];
    }

    return position;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    size_t index = GetIndex(gid, shape, strides, offset);
    out[gid] = a[index];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(
    const scalar_t* a,
    scalar_t* out,
    size_t size,
    CudaVec shape,
    CudaVec strides,
    size_t offset
) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size) {
        size_t index = GetIndex(gid, shape, strides, offset);
        out[index] = a[gid];
    }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);

  EwiseSetitemKernel<<<dim.grid, dim.block>>>(
      a.ptr,
      out->ptr,
      a.size,
      VecToCuda(shape),
      VecToCuda(strides),
      offset
  );
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(
    scalar_t val,
    scalar_t* out,
    size_t size,
    CudaVec shape,
    CudaVec strides,
    size_t offset
) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size) {
        size_t index = GetIndex(gid, shape, strides, offset);
        out[index] = val;
    }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);

  ScalarSetitemKernel<<<dim.grid, dim.block>>>(
      val,
      out->ptr,
      size,
      VecToCuda(shape),
      VecToCuda(strides),
      offset
  );
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
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


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

#define EWISE_OP(NAME, EXPR)                                           \
__global__ void NAME##Kernel(                                         \
    const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {\
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                \
    if (gid < size) {                                                  \
        scalar_t x = a[gid];                                           \
        scalar_t y = b[gid];                                           \
        out[gid] = (EXPR);                                             \
    }                                                                 \
}                                                                     \
void NAME(const CudaArray& a, const CudaArray& b, CudaArray* out) {    \
    CudaDims dim = CudaOneDim(out->size);                              \
    NAME##Kernel<<<dim.grid, dim.block>>>(                             \
        a.ptr, b.ptr, out->ptr, out->size);                            \
}

#define SCALAR_OP(NAME, EXPR)                                          \
__global__ void NAME##Kernel(                                         \
    const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {     \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                \
    if (gid < size) {                                                  \
        scalar_t x = a[gid];                                           \
        out[gid] = (EXPR);                                             \
    }                                                                 \
}                                                                     \
void NAME(const CudaArray& a, scalar_t val, CudaArray* out) {         \
    CudaDims dim = CudaOneDim(out->size);                              \
    NAME##Kernel<<<dim.grid, dim.block>>>(                             \
        a.ptr, val, out->ptr, out->size);                              \
}

#define UNARY_OP(NAME, EXPR)                                           \
__global__ void NAME##Kernel(                                         \
    const scalar_t* a, scalar_t* out, size_t size) {                   \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                \
    if (gid < size) {                                                  \
        scalar_t x = a[gid];                                           \
        out[gid] = (EXPR);                                             \
    }                                                                 \
}                                                                     \
void NAME(const CudaArray& a, CudaArray* out) {                       \
    CudaDims dim = CudaOneDim(out->size);                              \
    NAME##Kernel<<<dim.grid, dim.block>>>(                             \
        a.ptr, out->ptr, out->size);                                   \
}

EWISE_OP(EwiseMul, x * y)
SCALAR_OP(ScalarMul, x * val)

EWISE_OP(EwiseDiv, x / y)
SCALAR_OP(ScalarDiv, x / val)

SCALAR_OP(ScalarPower, powf(x, val))

EWISE_OP(EwiseMaximum, fmaxf(x, y))
SCALAR_OP(ScalarMaximum, fmaxf(x, val))

EWISE_OP(EwiseEq, x == y ? 1.0f : 0.0f)
SCALAR_OP(ScalarEq, x == val ? 1.0f : 0.0f)

EWISE_OP(EwiseGe, x >= y ? 1.0f : 0.0f)
SCALAR_OP(ScalarGe, x >= val ? 1.0f : 0.0f)

UNARY_OP(EwiseLog, logf(x))
UNARY_OP(EwiseExp, expf(x))
UNARY_OP(EwiseTanh, tanhf(x))

// Added for FP4 quantization (see python/needle/fp4/quantize.py).
UNARY_OP(EwiseFloor, floorf(x))
UNARY_OP(EwiseAbs, fabsf(x))

#undef EWISE_OP
#undef SCALAR_OP
#undef UNARY_OP

__global__ void MatmulKernel(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* out,
    uint32_t M,
    uint32_t N,
    uint32_t P
) {
    __shared__ scalar_t As[TILE][TILE];
    __shared__ scalar_t Bs[TILE][TILE];

    uint32_t row = blockIdx.y * TILE + threadIdx.y;
    uint32_t col = blockIdx.x * TILE + threadIdx.x;

    scalar_t sum = 0.0f;

    for (uint32_t t = 0; t < (N + TILE - 1) / TILE; t++) {
        uint32_t a_col = t * TILE + threadIdx.x;
        uint32_t b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < N) {
            As[threadIdx.y][threadIdx.x] = a[row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < N && col < P) {
            Bs[threadIdx.y][threadIdx.x] = b[b_row * P + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (uint32_t k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < P) {
        out[row * P + col] = sum;
    }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 block(TILE, TILE);

  dim3 grid(
      (P + TILE - 1) / TILE,
      (M + TILE - 1) / TILE
  );

  MatmulKernel<<<grid, block>>>(
      a.ptr,
      b.ptr,
      out->ptr,
      M,
      N,
      P
  );
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(
    const scalar_t* a,
    scalar_t* out,
    size_t out_size,
    size_t reduce_size
) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < out_size) {
        size_t start = gid * reduce_size;

        scalar_t max_val = a[start];

        for (size_t j = 1; j < reduce_size; j++) {
            max_val = fmaxf(max_val, a[start + j]);
        }

        out[gid] = max_val;
    }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);

  ReduceMaxKernel<<<dim.grid, dim.block>>>(
      a.ptr,
      out->ptr,
      out->size,
      reduce_size
  );
  /// END SOLUTION
}

__global__ void ReduceSumKernel(
    const scalar_t* a,
    scalar_t* out,
    size_t out_size,
    size_t reduce_size
) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < out_size) {
        size_t start = gid * reduce_size;

        scalar_t sum = 0.0f;

        for (size_t j = 0; j < reduce_size; j++) {
            sum += a[start + j];
        }

        out[gid] = sum;
    }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);

  ReduceSumKernel<<<dim.grid, dim.block>>>(
      a.ptr,
      out->ptr,
      out->size,
      reduce_size
  );
  /// END SOLUTION
}


////////////////////////////////////////////////////////////////////////////////
// NVFP4 (Stage 4): packed 4-bit storage, quantize+pack, dequantize, GEMV
//
// Same storage layout and arithmetic as the CPU versions in
// ndarray_backend_cpu.cc (read those first: they are the step-by-step
// versions). A (rows, cols) matrix is blocked along cols, zero-padded to
// padded_cols = multiple of 16:
//   packed: rows * padded_cols / 2 bytes (element 2i low nibble, 2i+1 high)
//   scales: rows * padded_cols / 16 E4M3 bytes
////////////////////////////////////////////////////////////////////////////////

#define FULL_MASK 0xffffffffu
#define WARP_SIZE 32

// Stop with a readable error if the last kernel launch failed.
#define CUDA_CHECK_LAUNCH()                                                   \
  do {                                                                        \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); \
  } while (0)

/** Byte buffer on the GPU for packed codes and E4M3 scales. */
struct CudaByteArray {
  CudaByteArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size == 0 ? 1 : size);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaByteArray() { cudaFree(ptr); }
  uint8_t* ptr;
  size_t size;
};

// ---- (a) quantize + pack ----------------------------------------------------

__global__ void AbsMaxKernel(const scalar_t* a, size_t size, float* out) {
  /**
   * out = max |a| over the whole array. Each thread takes a strided slice,
   * the warp combines its 32 values with __shfl_xor_sync, and one lane per
   * warp publishes the result with atomicMax. (For nonnegative floats the
   * IEEE bit pattern orders like the value, so atomicMax on the bits works.)
   */
  float m = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
    m = fmaxf(m, fabsf(a[i]));
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    m = fmaxf(m, __shfl_xor_sync(FULL_MASK, m, offset));
  if (threadIdx.x % WARP_SIZE == 0) atomicMax((int*)out, __float_as_int(m));
}

__global__ void Nvfp4QuantizePackKernel(const scalar_t* a, const scalar_t* rand, size_t rows,
                                        size_t cols, size_t cols_pad, const float* tensor_scale,
                                        uint8_t* packed, uint8_t* scales) {
  /**
   * One thread per element of the padded (rows, cols_pad) matrix. Because
   * cols_pad is a multiple of 16 and warps start at multiples of 32, every
   * block of 16 elements is exactly one half-warp, so:
   *   - the block amax is a __shfl_xor_sync reduction with offsets 8, 4, 2, 1
   *     (which never crosses between the two half-warps);
   *   - each even thread fetches its odd neighbour's code with
   *     __shfl_down_sync and writes the packed byte.
   * All 32 threads of a warp must reach the shuffles, so threads past the end
   * keep going with value 0 and only skip the writes.
   */
  using namespace nvfp4;
  size_t g = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = g < rows * cols_pad;
  size_t r = g / cols_pad, c = g % cols_pad;
  bool real = active && c < cols;                         // not padding
  float x = real ? a[r * cols + c] : 0.0f;
  float ts = *tensor_scale;

  float block_amax = fabsf(x);
  for (int offset = BLOCK / 2; offset > 0; offset /= 2)
    block_amax = fmaxf(block_amax, __shfl_xor_sync(FULL_MASK, block_amax, offset));

  float bs = block_scale(block_amax, ts);
  float scale = mul_rn(bs, ts);
  float v = scale > 0.0f ? div_rn(x, scale) : 0.0f;
  float u = (rand != nullptr && real) ? rand[r * cols + c] : 0.0f;
  int code = e2m1_encode(v, rand != nullptr, u);
  int next_code = __shfl_down_sync(FULL_MASK, code, 1);   // the odd neighbour's code

  if (active && g % 2 == 0) packed[g / 2] = (uint8_t)(code | (next_code << 4));
  if (active && g % BLOCK == 0) scales[g / BLOCK] = encode_e4m3(bs);
}

float Nvfp4QuantizePack(const CudaArray& a, size_t rows, size_t cols, const CudaArray* rand,
                        CudaByteArray* packed, CudaByteArray* scales) {
  /** Quantize a compact (rows, cols) matrix to packed NVFP4; returns the tensor scale. */
  size_t cols_pad = nvfp4::padded_cols(cols);
  float* d_amax;   // amax, then (overwritten) tensor scale, kept on the device
  cudaMalloc(&d_amax, sizeof(float));
  cudaMemset(d_amax, 0, sizeof(float));
  AbsMaxKernel<<<256, BASE_THREAD_NUM>>>(a.ptr, rows * cols, d_amax);
  CUDA_CHECK_LAUNCH();

  float amax;
  cudaMemcpy(&amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost);
  float ts = nvfp4::tensor_scale_from_amax(amax);
  cudaMemcpy(d_amax, &ts, sizeof(float), cudaMemcpyHostToDevice);

  CudaDims dim = CudaOneDim(rows * cols_pad);
  Nvfp4QuantizePackKernel<<<dim.grid, dim.block>>>(a.ptr, rand ? rand->ptr : nullptr, rows, cols,
                                                   cols_pad, d_amax, packed->ptr, scales->ptr);
  CUDA_CHECK_LAUNCH();
  cudaFree(d_amax);
  return ts;
}

// ---- dequantize -------------------------------------------------------------

__global__ void Nvfp4DequantizeKernel(const uint8_t* packed, const uint8_t* scales, float ts,
                                      size_t rows, size_t cols, size_t cols_pad, scalar_t* out) {
  using namespace nvfp4;
  size_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= rows * cols) return;
  size_t r = g / cols, c = g % cols, i = r * cols_pad + c;
  uint8_t byte = packed[i / 2];
  int code = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
  out[g] = mul_rn(e2m1_value(code), mul_rn(decode_e4m3(scales[i / BLOCK]), ts));
}

void Nvfp4Dequantize(const CudaByteArray& packed, const CudaByteArray& scales, float ts,
                     size_t rows, size_t cols, CudaArray* out) {
  CudaDims dim = CudaOneDim(rows * cols);
  Nvfp4DequantizeKernel<<<dim.grid, dim.block>>>(packed.ptr, scales.ptr, ts, rows, cols,
                                                 nvfp4::padded_cols(cols), out->ptr);
  CUDA_CHECK_LAUNCH();
}

// ---- (b) dequantize + GEMV --------------------------------------------------

__global__ void Nvfp4GemvKernel(const uint8_t* packed, const uint8_t* scales, float ts,
                                const scalar_t* x, scalar_t* y, size_t rows, size_t cols,
                                size_t cols_pad) {
  /**
   * y = W x, one warp per output row. Lane l reads blocks l, l+32, ...: each
   * block is 8 bytes of codes (one 64-bit load; neighbouring lanes read
   * neighbouring 8-byte chunks, so a warp reads 256 contiguous bytes at once)
   * plus one scale byte. Codes are turned into values with a 16-entry table in
   * shared memory, products are accumulated in FP32, and the 32 lane sums are
   * combined with a __shfl_xor_sync butterfly. Same order as the CPU version.
   */
  using namespace nvfp4;
  __shared__ float table[16];
  if (threadIdx.x < 16) table[threadIdx.x] = e2m1_value(threadIdx.x);
  __syncthreads();

  size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  if (row >= rows) return;                      // whole warps exit together

  size_t num_blocks = cols_pad / BLOCK;
  const uint8_t* row_codes = packed + row * (cols_pad / 2);
  float acc = 0.0f;
  for (size_t b = lane; b < num_blocks; b += WARP_SIZE) {
    uint2 bytes8 = *reinterpret_cast<const uint2*>(row_codes + b * 8);  // 16 codes
    float scale = mul_rn(decode_e4m3(scales[row * num_blocks + b]), ts);
    float s = 0.0f;
    for (int k = 0; k < 8; k++) {
      uint32_t word = k < 4 ? bytes8.x : bytes8.y;
      uint32_t byte = (word >> (8 * (k % 4))) & 0xFF;     // little-endian: byte k
      size_t c = b * BLOCK + 2 * k;
      float x0 = c < cols ? x[c] : 0.0f, x1 = c + 1 < cols ? x[c + 1] : 0.0f;
      s = add_rn(s, mul_rn(table[byte & 0xF], x0));       // element 2k (low nibble)
      s = add_rn(s, mul_rn(table[byte >> 4], x1));        // element 2k+1 (high nibble)
    }
    acc = add_rn(acc, mul_rn(s, scale));
  }
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    acc = add_rn(acc, __shfl_xor_sync(FULL_MASK, acc, offset));
  if (lane == 0) y[row] = acc;
}

__global__ void GemvFp32Kernel(const scalar_t* w, const scalar_t* x, scalar_t* y, size_t rows,
                               size_t cols) {
  /** Baseline FP32 y = W x with the same structure: one warp per row, lanes stride the row. */
  size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  if (row >= rows) return;
  const scalar_t* w_row = w + row * cols;
  float acc = 0.0f;
  if (cols % 4 == 0) {   // 16-byte loads when every row starts 16-byte aligned
    const float4* w4 = reinterpret_cast<const float4*>(w_row);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (size_t i = lane; i < cols / 4; i += WARP_SIZE) {
      float4 a = w4[i], b = x4[i];
      acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
  } else {
    for (size_t i = lane; i < cols; i += WARP_SIZE) acc += w_row[i] * x[i];
  }
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    acc += __shfl_xor_sync(FULL_MASK, acc, offset);
  if (lane == 0) y[row] = acc;
}

// One warp per row: blocks of 256 threads cover 8 rows.
static dim3 GemvGrid(size_t rows) {
  return dim3((rows * WARP_SIZE + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM);
}

void Nvfp4Gemv(const CudaByteArray& packed, const CudaByteArray& scales, float ts,
               const CudaArray& x, CudaArray* y, size_t rows, size_t cols) {
  Nvfp4GemvKernel<<<GemvGrid(rows), BASE_THREAD_NUM>>>(packed.ptr, scales.ptr, ts, x.ptr, y->ptr,
                                                       rows, cols, nvfp4::padded_cols(cols));
  CUDA_CHECK_LAUNCH();
}

void GemvFp32(const CudaArray& w, const CudaArray& x, CudaArray* y, size_t rows, size_t cols) {
  GemvFp32Kernel<<<GemvGrid(rows), BASE_THREAD_NUM>>>(w.ptr, x.ptr, y->ptr, rows, cols);
  CUDA_CHECK_LAUNCH();
}

// ---- benchmarking helpers ---------------------------------------------------

template <typename Launch>
float TimeKernelMs(Launch launch, int warmup, int iters) {
  /**
   * Average milliseconds per launch, measured with CUDA events on the GPU
   * itself, after `warmup` untimed launches (first launches pay one-time
   * costs). The final event synchronize makes sure all work has finished.
   */
  for (int i = 0; i < warmup; i++) launch();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) launch();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  CUDA_CHECK_LAUNCH();
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}

float BenchNvfp4Gemv(const CudaByteArray& packed, const CudaByteArray& scales, float ts,
                     const CudaArray& x, CudaArray* y, size_t rows, size_t cols, int warmup,
                     int iters) {
  return TimeKernelMs([&] { Nvfp4Gemv(packed, scales, ts, x, y, rows, cols); }, warmup, iters);
}

float BenchGemvFp32(const CudaArray& w, const CudaArray& x, CudaArray* y, size_t rows,
                    size_t cols, int warmup, int iters) {
  return TimeKernelMs([&] { GemvFp32(w, x, y, rows, cols); }, warmup, iters);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  // NVFP4 (Stage 4)
  py::class_<CudaByteArray>(m, "ByteArray")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaByteArray::size);
  m.def("bytes_to_numpy", [](const CudaByteArray& a) {
    py::array_t<uint8_t> out(a.size);
    cudaError_t err = cudaMemcpy(out.mutable_data(), a.ptr, a.size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return out;
  });
  m.def("bytes_from_numpy", [](py::array_t<uint8_t> a, CudaByteArray* out) {
    cudaError_t err = cudaMemcpy(out->ptr, a.request().ptr, out->size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });
  m.def("nvfp4_quantize_pack", Nvfp4QuantizePack, py::arg("a"), py::arg("rows"),
        py::arg("cols"), py::arg("rand").none(true), py::arg("packed"), py::arg("scales"));
  m.def("nvfp4_dequantize", Nvfp4Dequantize);
  m.def("nvfp4_gemv", Nvfp4Gemv);
  m.def("gemv_fp32", GemvFp32);
  m.def("bench_nvfp4_gemv", BenchNvfp4Gemv);
  m.def("bench_gemv_fp32", BenchGemvFp32);
  m.def("synchronize", [] { cudaDeviceSynchronize(); });
  m.def("device_info", [] {
    // Peak DRAM bandwidth = 2 transfers per clock (DDR) * memory clock * bus width.
    int dev, clock_khz, bus_bits;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    py::dict info;
    info["name"] = std::string(prop.name);
    info["peak_bandwidth_gb_s"] = 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
    info["memory_gb"] = prop.totalGlobalMem / 1e9;
    return info;
  });
}
