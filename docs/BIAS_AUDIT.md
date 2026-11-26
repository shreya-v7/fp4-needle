# Bias audit of the training experiments

Each row is a way the FP32 vs FP4 comparison could be unfair, what the code does about it, and how
that is checked. "Test" names a test in `tests/fp4`.

| Risk | What the code does | Checked by |
|---|---|---|
| Configurations start from different weights | `np.random.seed(seed)` right before building the model, for every configuration | Test `test_every_config_starts_from_the_same_weights` |
| Configurations see different batches | The data order is re-seeded at the start of every epoch (`seed * 1000 + epoch`), before the loader shuffles | Code review of `apps/fp4_experiment.py`; the shuffle is the first draw after the re-seed |
| Stochastic rounding shifts the data stream | SR has its own generator (`quantize.seed_sr`), restarted per run, so it never consumes `np.random` | Test `test_stochastic_rounding_does_not_touch_numpy_global_stream` |
| Fake quantization is not what NVFP4 hardware computes | Every value is rounded exactly as the NVFP4 spec says; a NumPy reference written from the spec is the ground truth, and every needle path matches it bit for bit | `tests/fp4/test_reference.py`, `test_fake_quant.py`, `test_kernels.py`. Remaining difference: real hardware sums FP4 products in its own order and precision, so the last bits of a matmul can differ |
| Results differ between CPU and GPU | Fake quantization uses the same float32 operations in the same order on every device | The same tests run on `cpu`, `cpu_numpy` and `cuda` (the CUDA ones need a GPU) |
| The learning rate was tuned for FP32 only (0.05, momentum 0.9) | Kept fixed for all configurations; round-to-nearest might simply need a smaller step | Learning-rate sweep (0.01, 0.05, 0.1) in `docs/PREREGISTRATION.md` |
| Short training makes slow learning look like failure | Main run trains 20 epochs instead of 3 | H3 in the pre-registration compares the gap at epoch 3 and epoch 20 |
| One noisy epoch decides the result | Metric A is the mean test accuracy of the last 3 epochs | Pre-registered |
| Picking the seeds or the metric after seeing results | Seeds 0 to 4, metric and pass/fail thresholds fixed and committed before running | `docs/PREREGISTRATION.md` (commit `c79893b`, before any run) |
| A diverged run silently disappears | The runner stops at a non-finite loss; its remaining epochs count as chance (10%) | `apps/fp4_stats.py` |
| "Dgrad" mixes two effects (rounding dY and rounding W) | `dgrad-dy` rounds only dY, `dgrad-w` only W | H2 in the pre-registration; tests check both against the reference |
| The test set is used to choose settings | No hyperparameter was chosen on the test set; the sweeps are reported, not used to pick a setting | By construction (there is no tuning step) |
| Evaluation differs between configurations | All are evaluated on the same 10,000 CIFAR-10 test images. Configurations with Fprop in FP4 are also evaluated with an FP4 forward pass, which is how such a model would be deployed | `evaluate` in `apps/fp4_experiment.py` |
| Small effects at small scale are over-read | Claims are stated for an MLP on CIFAR-10 and MNIST only; the report does not extrapolate to large models | Discussion section of the report |

Known limits that are not fixed, only reported: one architecture family (MLPs), one optimizer (SGD with
momentum), 20 epochs at most, and simulated (fake) FP4 arithmetic for training.
