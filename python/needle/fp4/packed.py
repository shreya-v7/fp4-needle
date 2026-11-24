"""Real 4-bit storage: an NVFP4 matrix packed in device memory (Stage 4).

Fake quantization (quantize.py) keeps every value in float32, so it can only
measure what FP4 does to accuracy. This module stores weights the way NVFP4
hardware would: two 4-bit codes per byte plus one E4M3 byte per 16 values,
and multiplies by a vector directly from that packed form. That is what saves
memory and time when generating text (batch size 1, so every step is a GEMV).

    W = NVFP4Matrix.quantize(weight)      # weight: NDArray (rows, cols) on cpu or cuda
    y = W.gemv(x)                         # x: NDArray (cols,)  ->  y: NDArray (rows,)
    W.nbytes, W.dense_nbytes              # 4-bit storage vs float32 storage

The kernels live in the backends (Nvfp4QuantizePack / Nvfp4Gemv in
src/ndarray_backend_cpu.cc and src/ndarray_backend_cuda.cu) and are tested bit
for bit against reference.quantize_nvfp4 / reference.nvfp4_gemv.
"""
import numpy as np

from ..backend_ndarray.ndarray import NDArray
from .reference import NVFP4_BLOCK_SIZE, NVFP4Tensor, decode_e4m3


def _padded(cols: int) -> int:
    return -(-cols // NVFP4_BLOCK_SIZE) * NVFP4_BLOCK_SIZE


class NVFP4Matrix:
    """A (rows, cols) matrix stored in packed NVFP4, blocked along cols.

    For y = W x the reduction axis is cols, so each row's blocks of 16 are what
    the GEMV multiplies with x.

    Attributes:
        shape: (rows, cols)
        device: the needle device (cpu or cuda) holding the bytes
        packed: device ByteArray, rows * padded_cols / 2 bytes of 4-bit codes
        scales: device ByteArray, rows * padded_cols / 16 bytes of E4M3 scales
        tensor_scale: the per-tensor FP32 scale (a Python float)
    """

    def __init__(self, shape, device, packed, scales, tensor_scale):
        self.shape = tuple(shape)
        self.device = device
        self.packed = packed
        self.scales = scales
        self.tensor_scale = tensor_scale

    @staticmethod
    def _check_device(device):
        if not hasattr(device, "nvfp4_quantize_pack"):
            raise NotImplementedError(
                f"packed NVFP4 needs the cpu or cuda backend, not {device}")

    @classmethod
    def quantize(cls, a: NDArray, stochastic: bool = False, rand: NDArray = None):
        """Quantize a 2-D NDArray to packed NVFP4 on its own device.

        With stochastic=True the elements use stochastic rounding with uniforms
        `rand` (same shape as a; drawn with a.device.rand if omitted).
        """
        assert a.ndim == 2, "NVFP4Matrix stores 2-D matrices"
        device = a.device
        cls._check_device(device)
        rows, cols = a.shape
        cols_pad = _padded(cols)
        packed = device.ByteArray(rows * cols_pad // 2)
        scales = device.ByteArray(rows * cols_pad // NVFP4_BLOCK_SIZE)
        if stochastic and rand is None:
            rand = device.rand(rows, cols)
        rand_handle = rand.compact()._handle if stochastic else None
        tensor_scale = device.nvfp4_quantize_pack(a.compact()._handle, rows, cols, rand_handle,
                                                  packed, scales)
        return cls((rows, cols), device, packed, scales, tensor_scale)

    def gemv(self, x: NDArray) -> NDArray:
        """y = W x, computed directly from the packed codes (no float copy of W)."""
        rows, cols = self.shape
        assert x.shape == (cols,), f"x must have shape ({cols},), got {x.shape}"
        y = NDArray.make((rows,), device=self.device)
        self.device.nvfp4_gemv(self.packed, self.scales, self.tensor_scale,
                               x.compact()._handle, y._handle, rows, cols)
        return y

    def dequantize(self) -> NDArray:
        """Expand back to a float32 (rows, cols) NDArray."""
        out = NDArray.make(self.shape, device=self.device)
        self.device.nvfp4_dequantize(self.packed, self.scales, self.tensor_scale,
                                     self.shape[0], self.shape[1], out._handle)
        return out

    # ---- memory accounting ----------------------------------------------------

    @property
    def nbytes(self) -> int:
        """Bytes used: 4-bit codes + 1-byte block scales + the 4-byte tensor scale."""
        return self.packed.size + self.scales.size + 4

    @property
    def dense_nbytes(self) -> int:
        """Bytes the same matrix takes in float32."""
        return self.shape[0] * self.shape[1] * 4

    # ---- for tests: copy the raw bytes to the host ------------------------------

    def to_reference(self) -> NVFP4Tensor:
        """The same data as a reference.NVFP4Tensor (so tests can compare bytes)."""
        rows, cols = self.shape
        cols_pad = _padded(cols)
        packed = self.device.bytes_to_numpy(self.packed).reshape(rows, cols_pad // 2)
        scale_bits = self.device.bytes_to_numpy(self.scales).reshape(rows, -1)
        return NVFP4Tensor(packed, decode_e4m3(scale_bits), np.float32(self.tensor_scale),
                           (rows, cols), axis=1)
