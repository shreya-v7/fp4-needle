"""Gradient-to-noise ratio for FP4 training (the sqrt(3) threshold of Chmiel et al., 2025).

    r = ||g|| / ||g_q - g||

g is the gradient with exact (FP32) backward matmuls, g_q the one actually used
for training; both use the same forward pass (forward quantization kept), so
g_q - g is only the backward-pass quantization noise. With zero-mean noise of
per-coordinate std sigma_q over d weights, ||g_q - g|| ~ sigma_q * sqrt(d), so r
is the paper's ||grad|| / (sigma_q sqrt(d)); training stalls once r < sqrt(3).
"""
import math

import numpy as np

from .config import QuantConfig

SQRT3 = math.sqrt(3.0)


def _configured(model):
    mods = [model] + list(model._children())
    return [m for m in mods if getattr(m, "config", None) is not None]


def set_config(model, make):
    """Replace every module's config with make(old_config); returns the old configs."""
    mods = _configured(model)
    old = [m.config for m in mods]
    for m in mods:
        m.config = make(m.config)
    return list(zip(mods, old))


def restore(saved):
    for m, c in saved:
        m.config = c


def exact_backward(cfg: QuantConfig) -> QuantConfig:
    return QuantConfig(fprop=cfg.fprop)


def _grads(model, loss_fn):
    for p in model.parameters():
        p.grad = None
    loss_fn().backward()
    return [p.grad.numpy().astype(np.float64) if p.grad is not None else None
            for p in model.parameters()]


def grad_noise_ratio(model, loss_fn):
    """r for one batch; loss_fn() runs the forward pass and returns the loss."""
    saved = set_config(model, exact_backward)
    try:
        g = _grads(model, loss_fn)
    finally:
        restore(saved)
    gq = _grads(model, loss_fn)
    for p in model.parameters():
        p.grad = None
    sig = sum(float(np.sum(a ** 2)) for a in g if a is not None)
    noise = sum(float(np.sum((b - a) ** 2)) for a, b in zip(g, gq) if a is not None)
    return math.sqrt(sig) / math.sqrt(noise) if noise > 0 else float("inf")
