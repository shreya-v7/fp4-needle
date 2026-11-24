"""FP4 (NVFP4) fake-quantized matrix multiply for linear layers."""

from ..autograd import Tensor, TensorOp
from ..fp4.config import QuantConfig
from ..fp4.quantize import fake_quant


class FP4Matmul(TensorOp):
    """Y = X @ W where each of the three training matmuls can run in FP4.

    X has shape (batch, in) and W has shape (in, out), like needle's nn.Linear.
    "Fake quantization" means each operand is rounded to a value NVFP4 can
    represent (quantize + dequantize), then an ordinary FP32 matmul runs. This
    reproduces FP4's numerical effect on training exactly, without needing FP4
    matmul hardware.

    Each matmul blocks its operands along *its own* reduction axis, so W is
    quantized along `in` for Fprop but along `out` for Dgrad (see config.py).

    Backward uses the straight-through estimator: quantization is treated as
    the identity, so gradients are computed from the unquantized X and W unless
    the Dgrad / Wgrad switches quantize the operands of those matmuls.
    """

    def __init__(self, config: QuantConfig):
        self.config = config

    def compute(self, X, W):
        if self.config.fprop:
            X = fake_quant(X, axis=1)   # reduce over `in` = axis 1 of X
            W = fake_quant(W, axis=0)   # ... and axis 0 of W
        return X @ W

    def gradient(self, out_grad, node):
        cfg = self.config
        X, W = (t.realize_cached_data() for t in node.inputs)
        dY = out_grad.realize_cached_data()

        # Dgrad: dX = dY @ W^T reduces over `out` (axis 1 of both dY and W).
        # (When SR is on, its random numbers are drawn here first, then in Wgrad.)
        if cfg.dgrad:
            dY_q = fake_quant(dY, axis=1, stochastic=cfg.stochastic)
            W_q = fake_quant(W, axis=1)
            dX = dY_q @ W_q.permute((1, 0))
        else:
            dX = dY @ W.permute((1, 0))

        # Wgrad: dW = X^T @ dY reduces over the batch (axis 0 of both X and dY).
        if cfg.wgrad:
            X_q = fake_quant(X, axis=0)
            dY_q = fake_quant(dY, axis=0, stochastic=cfg.stochastic)
            dW = X_q.permute((1, 0)) @ dY_q
        else:
            dW = X.permute((1, 0)) @ dY

        # The gradients are not differentiated again, so plain constants suffice.
        return Tensor.make_const(dX), Tensor.make_const(dW)


def fp4_matmul(X, W, config: QuantConfig):
    return FP4Matmul(config)(X, W)
