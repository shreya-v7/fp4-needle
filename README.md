# needle-fp4

4-bit floating point (NVFP4) training and inference inside **needle**, the deep
learning framework we built in CMU 10-414/714 (Deep Learning Systems).
Everything is implemented in needle itself: no PyTorch, TensorFlow, JAX or cuBLAS in
project code; NumPy is used only for the reference quantizer, data loading and tests.

## Questions

1. **Training:** a linear layer runs three matmuls: Fprop `Y = X W`, Dgrad `dX = dY Wᵀ`
   and Wgrad `dW = Xᵀ dY`. Which one breaks training when it runs in NVFP4, and does
   stochastic rounding (SR) fix it?
2. **Inference:** how much memory and matrix-vector (GEMV) time does real 4-bit weight
   storage save?
3. **Stretch:** how does MXFP4 compare with NVFP4?

## Historical results from earlier runs

* **Round-to-nearest gradients are what break 4-bit training.** CIFAR-10 MLP, 20 epochs, 5 seeds
  (pre-registered in [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md), bias audit in
  [`docs/BIAS_AUDIT.md`](docs/BIAS_AUDIT.md)): Dgrad with round-to-nearest loses 4.5 points against FP32,
  and rounding only its gradient dY loses the same while rounding only its weight W loses 0.2. With all three
  matmuls in FP4, round-to-nearest collapses to 12% (FP32 52%).
* **Stochastic rounding fixes most of it, not all.** Dgrad + SR is within 0.4 points of FP32, but all three
  in FP4 with SR ends 6.1 points behind after 20 epochs (0.7 after 3). Quantizing W consistently for the
  forward and backward pass (NVIDIA's 16x16 tiles, Thinking Machines' SR re-quantization) does not help.
  MNIST is too easy to separate the configurations. Details: [`docs/results/extended/verdicts.md`](docs/results/extended/verdicts.md).
* **Packed NVFP4 weights are 7x smaller** (4.5 bits per weight) with 97.42% vs 97.54% MNIST
  accuracy. Our FP32 GEMV reaches 84% of the GPU's peak bandwidth; the NVFP4 GEMV kernel was
  reworked after its first GPU run and is being re-measured.
* **NVFP4 has about 0.8 to 0.9 dB less weight error than MXFP4** on our MNIST and CIFAR-10 MLPs, for 0.25 more
  bits per weight; both keep accuracy within 0.5 points ([`docs/results/mxfp4/`](docs/results/mxfp4)).

Details: [`docs/results/`](docs/results) and the report notebook.

## NVFP4 in one paragraph

Each element is FP4 E2M1 (1 sign, 2 exponent, 1 mantissa bit), so it is one of
`{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with a sign. Values are grouped into blocks of 16 along the
matmul's reduction axis. Every block has a scale stored in FP8 E4M3, and the whole tensor has
one FP32 scale:

```
tensor_scale = amax(tensor) / (6 * 448)
block_scale  = round_to_e4m3((amax(block) / 6) / tensor_scale)
value        = e2m1(code) * block_scale * tensor_scale
```

Two codes are packed per byte (element `2i` in the low nibble). A 4096 x 4096 weight takes
64 MiB in FP32 and 9 MiB in NVFP4. Rounding conventions are documented at the top of
[`python/needle/fp4/reference.py`](python/needle/fp4/reference.py).

## Two ways we use FP4

* **Fake quantization (training).** Every matmul operand is rounded to a value NVFP4 can
  represent, then an ordinary FP32 matmul runs. This reproduces FP4's effect on accuracy
  exactly but saves nothing. It is bit-identical on CPU and GPU (tested), so the training
  grids were run on CPU. (`fp4/quantize.py`, `ops/ops_fp4.py`, `nn/nn_fp4.py`)
* **Real 4-bit storage (inference).** Weights are packed into 4-bit codes plus E4M3 scale
  bytes, and our kernels multiply a vector directly from the packed bytes. This is what saves
  memory and time. (`fp4/packed.py`, `src/nvfp4.h`, the `Nvfp4*` kernels in both backends)

## Repository layout

```
python/needle/fp4/      reference.py (NumPy reference), quantize.py (NDArray fake quant),
                        config.py (QuantConfig), packed.py (NVFP4Matrix + GEMV),
                        formats.py (NVFP4 vs MXFP4, stretch goal)
python/needle/ops/ops_fp4.py   FP4Matmul Op: each training matmul can run in FP4
python/needle/nn/nn_fp4.py     nn.QuantizedLinear, a drop-in for nn.Linear
src/                    CPU and CUDA backends; nvfp4.h holds the NVFP4 arithmetic both share
apps/                   fp4_experiment.py (MNIST / CIFAR-10 training grid), plot_fp4.py, fp4_stats.py,
                        fp4_grad_error.py (gradient error and bias of each FP4 matmul),
                        fp4_inference.py (MNIST from packed weights), fp4_mxfp4.py (stretch)
benchmarks/bench_gemv.py  FP32 vs NVFP4 GEMV kernels
notebooks/fp4_experiments.ipynb  the report and the one-shot Colab run
docs/results/           plots, tables and CSVs from our runs
tests/fp4/              our tests (NumPy and pytest only)
```

## Running

**Colab report.** Open [the report notebook](notebooks/fp4_experiments.ipynb) in a fresh runtime.
Run all for the paired demo, FP4 tests, packed inference, GEMV timings and format comparison.
Set `FULL_STUDY = True` before starting to include the registered five-seed study and sensitivity
runs. Download the run-specific
evidence ZIP before disconnecting. Archived measurements above come from earlier executions.

**Terminal**, from the repo root (`--device cpu` works for every training command):

```bash
make lib
PYTHONPATH=python python3 -m pytest tests/fp4 -q
python3 apps/fp4_experiment.py --dataset cifar10 --configs all --epochs 3 --seed 0 --out results/cifar_seed0.csv
python3 apps/fp4_experiment.py --dataset cifar10 --configs weights --epochs 3 --seed 0 --out results/cifar_w_seed0.csv
python3 apps/plot_fp4.py results/cifar_seed*.csv --out-dir results/cifar --title "CIFAR-10 MLP" --prefix cifar
python3 apps/fp4_grad_error.py --dataset cifar10 --checkpoints 0 391 1173 --out results/grad_error_cifar.csv
bash scripts/run_extended.sh          # the pre-registered 20-epoch study (about 6 h on 4 CPU cores, resumable)
python3 apps/fp4_stats.py docs/results/extended --out docs/results/extended/verdicts.md
python3 benchmarks/bench_gemv.py --device cuda --sizes 1024 2048 4096 8192
python3 apps/fp4_inference.py --device cuda
```

`--configs grid` is FP32, Fprop, Fprop+Dgrad, Fprop+Dgrad+Wgrad (with SR where it matters),
`--configs ablation` puts one backward matmul in FP4 at a time, `--configs all` runs both, and
`--configs weights` runs all three in FP4 with W quantized consistently (16x16 tiles, or SR
re-quantization of the forward W), and `--only NAME ...` reruns selected configs. SR applies to the gradient operand dY only, so
"FP32 + SR" and "Fprop + SR" would equal their SR-off runs and are not trained twice.

## How to read the metrics

| metric | meaning | how to read it |
|---|---|---|
| train loss vs FP32 (%) | (loss of this config / loss of FP32) - 1 | the clearest FP4 signal; a gap that keeps growing means FP4 is hurting learning |
| test accuracy (%) | share of test images classified correctly | trust a difference larger than the spread across seeds, with the same sign in every seed |
| gradient error vs FP32 | distance of the FP4 weight gradient from the exact one, relative to its size | the error of the mean of 32 SR draws shows what is left after averaging: the bias |
| time_us, bandwidth_GB_s | GEMV time (CUDA events) and bytes moved / time | a GEMV is limited by memory traffic; at the same % of peak, NVFP4 would be up to 7x faster |
| rel_error | distance of the NVFP4 GEMV output from the FP32 one | the accuracy price of 4-bit weights for one random matrix (about 10%) |

## AI assistance

Much of the FP4 code was written with an AI coding assistant (Claude, via Claude Code) and
reviewed and tested by the team. Every such file or edited section starts with a one-line
`AI-assisted:` marker. See the AI usage statement in the report notebook.

## Contributing

* **Never push to `main`.** All changes reach `main` through a pull request.
* **Branch names must be `dev/<firstname>-<topic>`**, lowercase, for example
  `dev/shreya-stage2-fake-quant`. CI rejects pull requests from any other branch name.
* CI builds the CPU backend and runs `tests/fp4` on every pull request; it must pass before merging.

## Known issues inherited from the homework code

These are left as they were in our homework submissions:

* `NDArray.__getitem__` with a negative slice start (`a[-2:]`) raises an `AssertionError`:
  `process_slice` sets `start = shape[dim]` instead of `shape[dim] + start`.
* `apps/simple_ml.py` `nn_epoch` computes `ReLU(X * W1) * W1`; the second factor should be `W2`.
  We do not use this function.
* The CUDA matmul uses 4x4 tiles (16 threads per block): correct but slow.
