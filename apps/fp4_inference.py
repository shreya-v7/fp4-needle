# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Inference with real 4-bit weights: memory, accuracy and batch-1 latency.

The training experiments use fake quantization, which saves nothing. Here the
weights are actually stored packed (NVFP4Matrix: 4-bit codes + E4M3 scales)
and the model runs from them:

  1. train an FP32 MLP on MNIST (a few epochs),
  2. pack every linear layer's weight into NVFP4 (biases stay FP32),
  3. report weight memory per layer (FP32 vs NVFP4),
  4. test accuracy with the 4-bit weights vs FP32,
  5. batch-size-1 latency, the "generation" setting: one input vector at a
     time, so every layer is a GEMV (FP32 GEMV vs NVFP4 GEMV kernels).

It also prints the weight memory of the LSTM language model from apps/models.py
at the sizes we train, to show the saving on a bigger model.

    python3 apps/fp4_inference.py --device cuda
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.append(os.path.dirname(__file__))

import numpy as np

import needle as ndl
from needle import backend_ndarray as nd
from needle.fp4.config import QuantConfig
from needle.fp4.packed import NVFP4Matrix
from fp4_experiment import evaluate, load_mnist, make_mlp, to_device

MiB = 2 ** 20


def nvfp4_bytes(rows, cols):
    """Bytes of a (rows, cols) matrix in NVFP4: 4-bit codes, 1 scale byte per 16, 4-byte tensor scale."""
    cols_pad = -(-cols // 16) * 16
    return rows * cols_pad // 2 + rows * cols_pad // 16 + 4


def memory_table(named_params):
    """Print FP32 vs NVFP4 bytes per parameter. Matrices are packed; vectors (biases) stay FP32."""
    total32 = total4 = 0
    print(f"  {'parameter':28s} {'shape':>14s} {'FP32':>10s} {'NVFP4':>10s}")
    for name, shape in named_params:
        b32 = int(np.prod(shape)) * 4
        b4 = nvfp4_bytes(*shape) if len(shape) == 2 else b32
        total32, total4 = total32 + b32, total4 + b4
        print(f"  {name:28s} {str(shape):>14s} {b32 / MiB:9.3f}M {b4 / MiB:9.3f}M")
    print(f"  {'total':28s} {'':>14s} {total32 / MiB:9.3f}M {total4 / MiB:9.3f}M  "
          f"({total32 / total4:.2f}x smaller)")
    return total32, total4


class PackedMLP:
    """The trained MLP with every weight packed in NVFP4 (inference only)."""

    def __init__(self, model):
        linears = [m for m in model.modules if isinstance(m, ndl.nn.QuantizedLinear)]
        # needle stores Linear weights as (in, out); y = W x needs (out, in), i.e. W^T.
        self.weights = [NVFP4Matrix.quantize(l.weight.realize_cached_data().permute((1, 0)).compact())
                        for l in linears]
        self.biases = [l.bias.realize_cached_data() for l in linears]

    def forward_one(self, x):
        """Batch size 1: x is an NDArray of shape (784,); every layer is one NVFP4 GEMV."""
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = W.gemv(x) + b
            if i < len(self.weights) - 1:
                x = x.maximum(0.0)                       # ReLU
        return x

    def forward_batch(self, X):
        """A whole batch: expand the 4-bit weights to float once and use a matmul."""
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            X = X @ W.dequantize().permute((1, 0)) + b.reshape((1, W.shape[0])).broadcast_to(
                (X.shape[0], W.shape[0]))
            if i < len(self.weights) - 1:
                X = X.maximum(0.0)
        return X


class Fp32MLP:
    """The same MLP with FP32 weights laid out (out, in) like PackedMLP, for a fair
    batch-1 comparison: both sides transpose once up front and then only run GEMVs."""

    def __init__(self, model):
        linears = [m for m in model.modules if isinstance(m, ndl.nn.QuantizedLinear)]
        self.weights = [l.weight.realize_cached_data().permute((1, 0)).compact() for l in linears]
        self.biases = [l.bias.realize_cached_data() for l in linears]

    def forward_one(self, x):
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            y = nd.NDArray.make((W.shape[0],), device=x.device)
            x.device.gemv_fp32(W._handle, x._handle, y._handle, W.shape[0], W.shape[1])
            x = y + b
            if i < len(self.weights) - 1:
                x = x.maximum(0.0)
        return x


def latency_ms(fn, inputs, device):
    """Average wall-clock milliseconds per input (GPU work is synchronized each time)."""
    for x in inputs[:5]:
        fn(x)                                           # warm-up
    start = time.perf_counter()
    for x in inputs:
        fn(x)
        if device.name == "cuda":
            device.synchronize()                        # wait for the GPU to finish
    return (time.perf_counter() - start) / len(inputs) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--latency-samples", type=int, default=200)
    args = parser.parse_args()
    device = getattr(ndl, args.device)()
    if not device.enabled():
        parser.error(f"needle's {args.device} backend is not built (Colab: use a GPU runtime)")

    # 1. Train an FP32 MLP.
    train_set, test_set = load_mnist(args.data_dir, args.train_subset)
    np.random.seed(0)
    model = make_mlp(QuantConfig(), 256, device)
    opt = ndl.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        for batch in ndl.data.DataLoader(train_set, batch_size=128, shuffle=True):
            X, y = to_device(batch, device)
            opt.reset_grad()
            ndl.nn.SoftmaxLoss()(model(X), y).backward()
            opt.step()

    # 2-3. Pack the weights and compare memory.
    packed = PackedMLP(model)
    print("\nMNIST MLP weight memory:")
    linears = [m for m in model.modules if isinstance(m, ndl.nn.QuantizedLinear)]
    params = []
    for i, l in enumerate(linears):
        params += [(f"layer{i + 1}.weight", (l.out_features, l.in_features)),
                   (f"layer{i + 1}.bias", (l.out_features,))]
    memory_table(params)
    measured = sum(W.nbytes for W in packed.weights)
    print(f"  (measured packed weight bytes: {measured / MiB:.3f}M)")

    # 4. Test accuracy, FP32 vs 4-bit weights.
    _, acc32 = evaluate(model, test_set, device)
    correct = n = 0
    for batch in ndl.data.DataLoader(test_set, batch_size=1000, shuffle=False):
        X, y = to_device(batch, device)
        logits = packed.forward_batch(X.realize_cached_data().reshape((X.shape[0], 784)))
        correct += int((logits.numpy().argmax(axis=1) == y.numpy()).sum())
        n += y.shape[0]
    print(f"\nTest accuracy: FP32 weights {acc32 * 100:.2f}%   NVFP4 weights {correct / n * 100:.2f}%")

    # 5. Batch-1 latency: one image at a time, every layer a GEMV.
    images = test_set.images[: args.latency_samples].reshape(-1, 784)
    inputs = [nd.array(img, device=device) for img in images]
    dense = Fp32MLP(model)
    t32 = latency_ms(dense.forward_one, inputs, device)
    t4 = latency_ms(packed.forward_one, inputs, device)
    agree = np.mean([dense.forward_one(x).numpy().argmax() == packed.forward_one(x).numpy().argmax()
                     for x in inputs])
    print(f"Batch-1 latency per image: FP32 {t32:.3f} ms   NVFP4 {t4:.3f} ms   "
          f"({t32 / t4:.2f}x); same prediction on {agree * 100:.1f}% of images")
    print("  (MNIST layers are small, so per-call overhead dominates here; "
          "benchmarks/bench_gemv.py measures the kernels at 4096 x 4096.)")

    # Bigger model: the LSTM language model's weights (vocab 10000 as in PTB).
    print("\nLSTM language model (PTB, embedding 100, hidden 100, 1 layer) weight memory:")
    vocab, emb, hid = 10000, 100, 100
    memory_table([("embedding", (vocab, emb)), ("lstm.W_ih", (4 * hid, emb)),
                  ("lstm.W_hh", (4 * hid, hid)), ("lstm.bias_ih", (4 * hid,)),
                  ("lstm.bias_hh", (4 * hid,)), ("linear.weight", (vocab, hid)),
                  ("linear.bias", (vocab,))])


if __name__ == "__main__":
    main()
