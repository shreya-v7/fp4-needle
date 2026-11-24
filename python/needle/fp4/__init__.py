"""FP4 (NVFP4) quantization for needle.

Modules:
    reference: NumPy ground-truth quantizer used to test everything else.
    quantize:  the same fake quantization written with needle NDArray ops.
    config:    QuantConfig, which matmuls of a linear layer run in FP4.

The FP4 linear Op lives in needle.ops.ops_fp4 and the layer in
needle.nn.nn_fp4 (nn.QuantizedLinear).
"""
from . import reference, quantize, config
from .config import QuantConfig
