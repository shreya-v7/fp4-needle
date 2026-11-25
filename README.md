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
3. **Storage formats:** at about 4.5 bits per weight, which 4-bit format (NVFP4,
   MXFP4, NVINT4, INT4, NF4, with or without a Hadamard rotation) loses the least,
   on our models and on GPT-2? And does 4-bit training follow the published
   sqrt(3) gradient-to-noise rule?

## Status

| Stage | What | Status |
|---|---|---|
| 0 | Read the homework code, plan the changes | done |
| 1 | NumPy reference quantizer (`python/needle/fp4/reference.py`) + tests | done |
| 2 | Fake-quant FP4 linear Op and `nn.QuantizedLinear` | done (CPU); CUDA tests run on Colab |
| 3 | MNIST MLP experiments (FP32 / Fprop / +Dgrad / +Wgrad, with and without SR) | done on CPU (below); GPU runs via the notebook |
| 4 | Packed NVFP4 storage, quantize+pack and FP4 GEMV kernels (CPU + CUDA), GEMV benchmark, 4-bit inference | done; CPU bit-exact, CUDA tests and benchmark run on Colab |
| + | LSTM language model with FP4 matmuls (PTB) | done (below) |
| + | Paper experiments: format storage study, sqrt(3) monitor, mini-GPT | storage study done on CPU (below); sweeps run via `notebooks/fp4_paper_experiments.ipynb` |

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

Two 4-bit codes are packed per byte (element `2i` in the low nibble). A 4096 x 4096
weight matrix takes 64 MiB in FP32 and 9 MiB in NVFP4 (8 MiB of codes + 1 MiB of
scales), about 7x less. Our
rounding conventions (ties to even, stochastic rounding only on elements, zero
always encoded as `+0`) are documented at the top of
[`python/needle/fp4/reference.py`](python/needle/fp4/reference.py).

## Two ways we use FP4

* **Fake quantization (training experiments).** Every matmul operand is rounded to a
  value NVFP4 can represent, then an ordinary FP32 matmul runs. This reproduces FP4's
  effect on *accuracy* exactly, but stores everything in FP32 and adds work, so these
  runs are slower than FP32, not faster. (`fp4/quantize.py`, `ops/ops_fp4.py`)
* **Real 4-bit storage (inference).** Weights are packed into 4-bit codes plus E4M3
  scale bytes, and our CUDA kernel multiplies a vector directly from the packed bytes.
  This is what saves *memory* and *time*. (`fp4/packed.py`, `src/nvfp4.h`, the
  `Nvfp4*` kernels in `src/ndarray_backend_cuda.cu`)

## Repository layout

```
python/needle/          needle from our homeworks (hw0 to hw4 + transformer)
python/needle/fp4/      FP4 code: NumPy reference, NDArray fake quantization, QuantConfig,
                        packed.py (NVFP4Matrix: real 4-bit storage + GEMV),
                        formats.py (plug-in 4-bit formats, rotations), monitor.py (sqrt(3) test)
python/needle/ops/ops_fp4.py   FP4Matmul Op (each training matmul can run in FP4)
python/needle/nn/nn_fp4.py     nn.QuantizedLinear (drop-in for nn.Linear)
src/                    C++ (CPU) and CUDA backends, exposed with pybind11;
                        nvfp4.h holds the NVFP4 arithmetic both backends share
apps/                   homework models, plus fp4_experiment.py (MNIST), fp4_lstm_experiment.py
                        (PTB LSTM), fp4_gpt_experiment.py (mini-GPT), fp4_storage.py
                        (format study), fp4_grad_error.py, fp4_inference.py, plot_fp4.py, plot_paper.py
benchmarks/             bench_gemv.py: FP32 vs NVFP4 GEMV kernels
notebooks/              fp4_experiments.ipynb: tests + experiments on a Colab GPU;
                        fp4_paper_experiments.ipynb: the paper experiments (E1 to E4)
docs/results/           plots and summary tables from our runs
tests/hw4, hw4_extra/   course tests (need PyTorch and mugrade, see below)
tests/fp4/              our FP4 tests (NumPy and pytest only)
```

## Running on Google Colab

Open `notebooks/fp4_experiments.ipynb` from GitHub in Colab (File > Open notebook >
GitHub), pick a GPU runtime (Runtime > Change runtime type), and Run all. It clones
the repo into Drive, builds needle with `make` (CPU and CUDA backends), runs the tests,
the Stage 4 benchmark and every experiment. Set `BRANCH` in the first cell to try a
`dev/<name>-<topic>` branch. If the repo is private, add a `GITHUB_TOKEN` secret in
Colab (the setup cell explains how).

The course tests in `tests/hw4` and `tests/hw4_extra` compare against PyTorch
(already installed on Colab) and import `mugrade`:
`!pip3 install --upgrade --no-deps git+https://github.com/dlsys10714/mugrade.git`.
They also need the datasets in `data/` (not tracked in git); the notebook downloads
MNIST and PTB.

## Running the experiments

The easiest way is `notebooks/fp4_experiments.ipynb` on Colab with a GPU runtime: it
builds needle, runs the tests (including CUDA), and trains every configuration.
From a terminal in the repo root:

```bash
python3 apps/fp4_experiment.py --configs all --device cuda --epochs 3 --seed 0 --out results/mnist_seed0.csv
python3 apps/plot_fp4.py results/mnist_seed*.csv --out-dir results
python3 apps/fp4_lstm_experiment.py --configs all --device cuda --epochs 3 --seed 0 --out results/lstm_seed0.csv
python3 apps/plot_fp4.py results/lstm_seed*.csv --out-dir results/lstm --perplexity --title "PTB LSTM" --prefix lstm
python3 benchmarks/bench_gemv.py --device cuda --sizes 1024 2048 4096 8192
python3 apps/fp4_inference.py --device cuda
```

`--configs grid` is the project grid (FP32, Fprop, Fprop+Dgrad, Fprop+Dgrad+Wgrad,
with SR where it matters), `--configs ablation` puts one backward matmul in FP4 at a
time, and `--configs all` runs both, and `--only NAME ...` reruns selected configs.
`apps/fp4_grad_error.py` measures each FP4 matmul's gradient error against FP32. Stochastic rounding applies to the gradient
operand dY only, so "FP32 + SR" and "Fprop + SR" would equal their SR-off runs and
are not trained twice.

## How to read the metrics

**Training experiments (accuracy question)**

| metric | meaning | how to read it |
|---|---|---|
| train loss (smoothed) | cross-entropy on the training batches, averaged over recent steps | lower is better; curves that overlap mean FP4 did not hurt |
| train loss vs FP32 (%) | (loss of this config / loss of FP32) - 1 | the clearest FP4 signal: +3% means 3% higher loss than FP32 at the same step; a gap that keeps growing means FP4 is hurting learning |
| test accuracy (%) | share of test images (or next words) predicted correctly | differences smaller than the min-max bars across seeds are noise |
| test perplexity | exp(test loss), for the language model: how many words the model is "choosing between" | lower is better; compare against FP32 |
| gradient error vs FP32 (%) | the FP4 weight gradient's distance from the exact FP32 gradient, relative to its size | how much each FP4 matmul distorts learning; the dot (mean of 32 SR draws) shows what remains after averaging, i.e. the bias |
| zeros % (grad_error.csv) | gradient entries that FP4 rounded to exactly 0 | underflow: those weights get no update at all |

Rule of thumb: one seed and one quick epoch only show that things run. Trust a difference
when it is larger than the spread across seeds and has the same sign in every seed.

**Stage 4 benchmark (systems question)**

| metric | meaning | how to read it |
|---|---|---|
| time_us | average time of one y = W x (CUDA events, after warm-up) | lower is better; `speedup` = FP32 time / this time |
| weight_MiB | memory holding W | NVFP4 should be about 7x smaller (4.5 bits per value) |
| bytes_moved_MiB | data the kernel must read or write at least once (W, x, y) | a GEMV does about 2 floating-point operations per weight, so it is limited by memory traffic, not arithmetic |
| bandwidth_GB_s | bytes_moved / time | how fast the kernel streams memory |
| pct_of_peak | bandwidth as a share of the GPU's peak DRAM bandwidth | near 100% means the kernel is as fast as the memory allows; if NVFP4 reaches a similar % as FP32, the speedup approaches the 7x drop in bytes |
| rel_error | distance between the NVFP4 and FP32 GEMV outputs, relative to the FP32 output | the accuracy price of 4-bit weights for one matrix (about 10% for random Gaussian weights); fp4_inference.py shows what it means for a real model's accuracy |

## Results so far (CPU, MNIST)

784-256-256-10 MLP, every linear layer an `nn.QuantizedLinear`, 3 epochs, batch 128.
SGD is our homework optimizer (lr 0.05, momentum 0.9) with 3 seeds; Adam (lr 1e-3) has 1 seed.
Full tables: [`docs/results/cpu_sgd/summary.md`](docs/results/cpu_sgd/summary.md),
[`docs/results/cpu_adam/summary.md`](docs/results/cpu_adam/summary.md).

| config | SGD test acc (%) | SGD train loss | Adam test acc (%) | Adam train loss |
|---|---|---|---|---|
| FP32 | 94.90 ± 0.04 | 0.183 | 97.52 | 0.061 |
| Fprop | 94.75 ± 0.04 | 0.190 | 97.36 | 0.068 |
| Fprop + Dgrad | 94.74 ± 0.04 | 0.190 | 97.49 | 0.065 |
| Fprop + Dgrad + SR | 94.67 ± 0.05 | 0.188 | 97.57 | 0.060 |
| Fprop + Dgrad + Wgrad | 94.74 ± 0.18 | **0.197** | 97.43 | **0.071** |
| Fprop + Dgrad + Wgrad + SR | 94.79 ± 0.09 | 0.181 | 97.66 | 0.056 |

(train loss = mean of the last 100 steps; SR = stochastic rounding of the gradient dY)

What we see:

* **Nothing breaks on this model.** Every configuration ends within about 0.3 points of
  FP32 test accuracy. A small MLP trained for 3 epochs is a forgiving setting.
* **Fprop is the most expensive single matmul here:** about 3% higher training loss for
  the whole run, and the largest change in the gradients (24 to 47% relative difference
  in `docs/results/cpu_sgd/grad_error.png`), because it changes the function being trained.
* **Wgrad with round-to-nearest is where gradients get damaged:** once gradients shrink
  during training, 3 to 4.5% of weight-gradient entries round to exactly zero (FP4
  underflow), and all-FP4 round-to-nearest has the highest training loss with both
  optimizers.
* **Stochastic rounding fixes it:** with SR, all-FP4 training loss is back at (or below)
  FP32. In the gradient-error experiment a single SR gradient is noisier than
  round-to-nearest, but the average of 32 SR gradients is closer to FP32, i.e. SR trades
  bias for noise. It does not remove all bias, because X and W are still rounded to nearest.

**LSTM language model (CPU, PTB).** 1 layer, embedding and hidden size 64, the first 1000
lines of PTB (vocabulary 4865), 3 epochs, Adam (lr 3e-3), 3 seeds. Full table:
[`docs/results/cpu_lstm/summary.md`](docs/results/cpu_lstm/summary.md).

| config | test perplexity | train loss (last 100 steps) |
|---|---|---|
| FP32 | 664.7 ± 1.7 | 5.985 |
| Fprop | 671.2 ± 3.7 | 5.993 |
| Fprop + Dgrad | 674.4 ± 3.2 | 6.002 |
| Fprop + Dgrad + Wgrad | **651.3 ± 1.6** | 6.002 |
| Fprop + Dgrad + Wgrad + SR | 673.4 ± 2.6 | 5.999 |

* Again nothing breaks: every FP4 configuration is within about 1.5% of FP32 perplexity, and
  the training-loss gap to FP32 grows slowly (to 0.15 to 0.3% after 207 steps).
* All-FP4 with round-to-nearest has the *worst* training loss but the *best* test perplexity, in
  all 3 seeds. With this little data the model generalizes poorly (train perplexity about 370,
  test about 665), and the rounding of weight gradients (small entries become exactly zero)
  appears to act as a regularizer. With SR, which removes that bias, the effect disappears.
  This is worth checking with more data (`notebooks/fp4_experiments.ipynb` uses 2000 lines).

Real 4-bit weights (`apps/fp4_inference.py`, CPU): the MLP's weights shrink from 1.03 MiB
to 0.146 MiB (7.0x), and test accuracy with the packed NVFP4 weights is 94.01% vs 94.05%
in FP32, with the same prediction on 99% of images. The LSTM language model's weights
shrink 6.2x (rows of 100 values are padded to 112). GEMV timings need a GPU: CPU kernels
here are the readable reference versions, not optimized.

![MNIST grid](docs/results/cpu_sgd/mnist_grid.png)
![Gradient error](docs/results/cpu_sgd/grad_error.png)

## Paper experiments

`notebooks/fp4_paper_experiments.ipynb` runs four experiments on Colab (set `QUICK=False`
for the full sweeps); the last cell zips `results/paper`.

| | Experiment | Script |
|---|---|---|
| E1 | Storage: error of each 4-bit format with and without rotation, on our checkpoints and GPT-2, plus post-training quantization | `apps/fp4_storage.py` |
| E2 | sqrt(3) rule: gradient-to-noise ratio during training, and switching the backward pass to FP32 | `apps/fp4_gpt_experiment.py --monitor-every --switch-step` |
| E3 | Data size: LSTM on 2k, 10k and all PTB lines | `apps/fp4_lstm_experiment.py --max-lines` |
| E4 | Model size: mini-GPT on text8 at 0.1M, 0.8M, 4.7M parameters | `apps/fp4_gpt_experiment.py --size` |

CPU results for E1 are in [`docs/results/paper_cpu/`](docs/results/paper_cpu). On all 48
GPT-2 matmul weights (SQNR in dB, higher is better):

| Format | Bits per weight | No rotation | Hadamard 16 | Hadamard 128 |
|---|---|---|---|---|
| NVINT4 | 4.5 | 21.08 | 21.33 | 21.29 |
| NF4-g64 | 4.25 | 20.56 | 20.72 | 20.73 |
| NVFP4 | 4.5 | 20.47 | 20.42 | 20.43 |
| INT4-g32 | 4.5 | 19.96 | 20.30 | 20.28 |
| MXFP4 | 4.25 | 18.66 | 18.76 | 18.76 |
| INT4-g128 | 4.125 | 17.93 | 18.51 | 18.65 |

NVINT4 (integer levels with NVFP4's scales) beats NVFP4 on 98% of the matrices, and
INT4's error follows NMSE = crest factor / 588 (correlation 0.97). Both match published
work: Chen et al., "INT v.s. FP" (arXiv 2510.25602) and Castro, Egiazarian et al.,
"Bridging the Gap" (arXiv 2509.23202).

## AI assistance

Much of the FP4 code was written with an AI coding assistant (Claude, via Claude Code)
and reviewed and tested by the team. Every such file or edited section starts with a
one-line `AI-assisted:` marker. See the AI usage statement in our final report.

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
