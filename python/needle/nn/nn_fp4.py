"""nn.QuantizedLinear: a drop-in replacement for nn.Linear that trains in FP4."""

from needle import ops
from needle.fp4.config import QuantConfig
from .nn_basic import Linear


class QuantizedLinear(Linear):
    """nn.Linear whose matmuls are fake-quantized to NVFP4 according to `config`.

    Parameters are created exactly as in nn.Linear (same shapes, same
    initialization), so with QuantConfig() (everything off) it behaves exactly
    like nn.Linear. The bias add stays in FP32: it is not a matmul.
    """

    def __init__(self, in_features, out_features, bias=True, device=None,
                 dtype="float32", config: QuantConfig = QuantConfig()):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.config = config

    def forward(self, X):
        out = ops.fp4_matmul(X, self.weight, self.config)
        if self.bias is not None:  # same bias handling as nn.Linear.forward
            bias = self.bias.reshape((1, self.out_features))
            out = out + bias.broadcast_to(out.shape)
        return out
