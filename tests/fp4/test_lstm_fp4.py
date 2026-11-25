# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Tests for the FP4 option of needle's LSTM (config=QuantConfig(...)).

    python3 -m pytest tests/fp4/test_lstm_fp4.py -v
"""
import os
import sys

sys.path.append("./python")
sys.path.append("./apps")

import numpy as np
import pytest

import needle as ndl
from needle import backend_ndarray as nd
from needle.fp4 import reference as ref
from needle.fp4.config import FP32, FPROP, QuantConfig

DEVICES = [
    nd.cpu(),
    pytest.param(nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="no CUDA")),
]


def _cell_outputs(cell, X, h0, c0, device):
    Xt = ndl.Tensor(X, device=device)
    h, c = cell(Xt, (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
    (h.sum() + c.sum()).backward()
    return h.numpy(), c.numpy(), Xt.grad.numpy(), cell.W_ih.grad.numpy(), cell.W_hh.grad.numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_lstm_cell_with_fp32_config_equals_original(device):
    rng = np.random.default_rng(0)
    X, h0, c0 = (rng.standard_normal(s).astype(np.float32) for s in [(5, 12), (5, 20), (5, 20)])
    np.random.seed(1)
    original = ndl.nn.LSTMCell(12, 20, device=device)
    np.random.seed(1)
    fp4_off = ndl.nn.LSTMCell(12, 20, device=device, config=FP32)
    for a, b in zip(_cell_outputs(original, X, h0, c0, device), _cell_outputs(fp4_off, X, h0, c0, device)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_lstm_cell_fprop_quantizes_both_matmuls(device):
    """With Fprop on, the cell must equal the original cell run on inputs and weights
    that were fake-quantized beforehand (X and h along axis 1, W_ih and W_hh along axis 0)."""
    rng = np.random.default_rng(2)
    X, h0, c0 = (rng.standard_normal(s).astype(np.float32) for s in [(6, 40), (6, 24), (6, 24)])
    np.random.seed(3)
    cell = ndl.nn.LSTMCell(40, 24, device=device, config=FPROP)
    np.random.seed(3)
    plain = ndl.nn.LSTMCell(40, 24, device=device)
    plain.W_ih.data = ndl.Tensor(ref.fake_quant(plain.W_ih.numpy(), axis=0), device=device)
    plain.W_hh.data = ndl.Tensor(ref.fake_quant(plain.W_hh.numpy(), axis=0), device=device)
    Xq, hq = ref.fake_quant(X, axis=1), ref.fake_quant(h0, axis=1)

    h_fp4, c_fp4 = cell(ndl.Tensor(X, device=device),
                        (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
    h_ref, c_ref = plain(ndl.Tensor(Xq, device=device),
                         (ndl.Tensor(hq, device=device), ndl.Tensor(c0, device=device)))
    np.testing.assert_array_equal(h_fp4.numpy(), h_ref.numpy())
    np.testing.assert_array_equal(c_fp4.numpy(), c_ref.numpy())


def test_fp4_language_model_trains():
    """Smoke test: the experiment's FP4 LSTM language model (all matmuls in FP4 + SR)
    lowers its loss on a tiny repeating corpus."""
    from fp4_lstm_experiment import FP4LanguageModel
    device = nd.cpu()
    np.random.seed(4)
    cfg = QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True)
    model = FP4LanguageModel(vocab_size=12, embedding_size=16, hidden_size=16, num_layers=1,
                             config=cfg, device=device)
    opt = ndl.optim.Adam(model.parameters(), lr=0.02)
    tokens = np.tile(np.arange(12), 8).astype(np.float32)            # 0 1 2 ... 11 0 1 ...
    data = tokens[: 8 * 12].reshape(-1, 4)                           # (24, 4) like batchify
    losses = []
    for _ in range(25):
        X, y = ndl.data.get_batch(data, 0, 10, device=device, dtype="float32")
        logits, _ = model(X)
        loss = ndl.nn.SoftmaxLoss()(logits, y)
        opt.reset_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.numpy()))
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.5 * losses[0]
