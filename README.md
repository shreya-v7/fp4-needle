# needle-fp4

4-bit floating point (NVFP4) training and inference inside **needle**, the deep
learning framework we built in CMU 10-414/714 (Deep Learning Systems).
Everything is implemented in needle itself. There is no PyTorch, TensorFlow, or
JAX in project code; NumPy is used only for the reference quantizer and tests.

## Questions we are answering

1. **Research:** a linear layer runs three matmuls during training:
   Fprop `Y = X W`, Dgrad `dX = dY Wᵀ`, and Wgrad `dW = Xᵀ dY`.
   When each one runs in FP4, which one breaks training, and does stochastic
   rounding fix it?
2. **Systems:** how much memory and matrix-vector (GEMV) time does 4-bit weight
   storage save during generation?

## Status

| Stage | What | Status |
|---|---|---|
| 0 | Read the homework code, plan the changes | done |
| 1 | NumPy reference quantizer (`python/needle/fp4/reference.py`) + tests | in review |
| 2 | Fake-quant FP4 linear Op and `nn.QuantizedLinear` | not started |
| 3 | MNIST MLP experiments (FP32 / Fprop / +Dgrad / +Wgrad, with and without SR) | not started |
| 4 | CUDA quantize+pack and FP4 GEMV kernels, 4096x4096 benchmark | not started |

## NVFP4 in one paragraph

Each element is FP4 E2M1 (1 sign, 2 exponent, 1 mantissa bit), so it can only
be one of `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with a sign. Values are grouped into
blocks of 16 along the matmul's reduction axis. Every block has a scale stored
in FP8 E4M3, and the whole tensor has one FP32 scale:

```
tensor_scale = amax(tensor) / (6 * 448)
block_scale  = round_to_e4m3((amax(block) / 6) / tensor_scale)
value        = e2m1(code) * block_scale * tensor_scale
```

Two 4-bit codes are packed per byte (element `2i` in the low nibble). Our
rounding conventions (ties to even, stochastic rounding only on elements, zero
always encoded as `+0`) are documented at the top of
[`python/needle/fp4/reference.py`](python/needle/fp4/reference.py).

## Repository layout

```
python/needle/          needle from our homeworks (hw0 to hw4 + transformer)
python/needle/fp4/      FP4 code we are adding (reference quantizer so far)
src/                    C++ (CPU) and CUDA backends, exposed with pybind11
apps/                   models and training loops from the homeworks
tests/hw4, hw4_extra/   course tests (need PyTorch and mugrade, see below)
tests/fp4/              our FP4 tests (NumPy and pytest only)
```

## Running on Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/10714
!git clone https://github.com/shreya-v7/fp4-needle.git
%cd fp4-needle
!pip3 install pybind11
!make
%set_env PYTHONPATH ./python
%set_env NEEDLE_BACKEND nd
!python3 -m pytest tests/fp4 -v
```

Pick a GPU runtime (Runtime > Change runtime type) so `make` also builds the
CUDA backend. To try someone's branch: `!git fetch && git checkout dev/<name>-<topic>`.

The course tests in `tests/hw4` and `tests/hw4_extra` compare against PyTorch
(already installed on Colab) and import `mugrade`:
`!pip3 install --upgrade --no-deps git+https://github.com/dlsys10714/mugrade.git`.
They also need the datasets in `data/` (not tracked in git). MNIST is in our
Drive folder under `hw2/data`, so `!mkdir -p data && cp ../hw2/data/*.gz data/`.

## Contributing

* **Never push to `main`.** All changes reach `main` through a pull request.
* **Branch names must be `dev/<firstname>-<topic>`**, lowercase, for example
  `dev/shreya-stage2-fake-quant`. CI rejects pull requests from any other branch name.
* CI builds the CPU backend and runs `tests/fp4` on every pull request; it
  must pass before merging.
* Keep pull requests small (one stage or one piece of a stage) and say in the
  description how you tested it.

## Known issues inherited from the homework code

These are left as they were in our homework submissions:

* `NDArray.__getitem__` with a negative slice start (`a[-2:]`) raises an
  `AssertionError`: `process_slice` sets `start = shape[dim]` instead of
  `shape[dim] + start`.
* `apps/simple_ml.py` `nn_epoch` computes `ReLU(X * W1) * W1`; the second
  factor should be `W2`. We do not use this function.
* The CUDA matmul uses 4x4 tiles (16 threads per block): correct but slow.
