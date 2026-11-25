# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Which of a linear layer's three training matmuls run in FP4.

For Y = X @ W with X: (batch, in) and W: (in, out):
    Fprop  Y  = X  @ W     reduces over `in`:   X blocked along axis 1, W along axis 0
    Dgrad  dX = dY @ W^T   reduces over `out`:  dY along axis 1,        W along axis 1
    Wgrad  dW = X^T @ dY   reduces over batch:  X along axis 0,         dY along axis 0
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantConfig:
    """Switches for FP4 fake quantization of a linear layer.

    Attributes:
        fprop: quantize X and W in the forward matmul.
        dgrad: quantize dY and W in the input-gradient matmul.
        wgrad: quantize X and dY in the weight-gradient matmul.
        stochastic: use stochastic rounding for the gradient operand dY (in
            Dgrad and Wgrad). Weights and activations always use
            round-to-nearest, as in NVIDIA's NVFP4 training recipe: SR removes
            the bias that round-to-nearest adds to gradients, while SR on
            forward-pass tensors would only add noise.
        weights: how W is quantized for the two matmuls that use it.
            "1d": independently in each, blocks of 16 along its reduction axis,
                  so Fprop and Dgrad see slightly different weights (default);
            "2d": one scale per 16 x 16 tile, so both see the same weight
                  (NVIDIA's NVFP4 recipe);
            "sr": Dgrad re-quantizes the forward weight with stochastic
                  rounding, equal to it in expectation (Thinking Machines' recipe).
    """
    fprop: bool = False
    dgrad: bool = False
    wgrad: bool = False
    stochastic: bool = False
    weights: str = "1d"

    def __post_init__(self):
        assert self.weights in ("1d", "2d", "sr"), f"unknown weights mode {self.weights!r}"

    @property
    def name(self) -> str:
        """Short label used in experiment logs, e.g. 'fprop+dgrad+sr'."""
        parts = [p for p, on in (("fprop", self.fprop), ("dgrad", self.dgrad),
                                 ("wgrad", self.wgrad)) if on]
        label = "+".join(parts) if parts else "fp32"
        label += "+sr" if self.stochastic else ""
        return label + ("" if self.weights == "1d" else f"+w{self.weights}")


# The configurations from the project plan.
FP32 = QuantConfig()
FPROP = QuantConfig(fprop=True)
FPROP_DGRAD = QuantConfig(fprop=True, dgrad=True)
FPROP_DGRAD_WGRAD = QuantConfig(fprop=True, dgrad=True, wgrad=True)
