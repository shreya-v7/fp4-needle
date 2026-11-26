# Pre-registration: extended experiments

Written and committed before any of these runs started. The criteria below decide the verdicts; they
are not changed after seeing results. Every run is reported; none is excluded or repeated.

## Setup (fixed)

* CIFAR-10, all 50,000 training images, MLP with ReLU, hidden width 256, SGD (momentum 0.9), batch 128.
* Every configuration of a seed starts from the same weights and sees the same batches; stochastic
  rounding has its own random stream (`quantize.seed_sr`), restarted per run.
* **Metric A** for one run: mean test accuracy over its last 3 epochs (less sensitive to one noisy epoch
  than the final epoch alone). A run whose loss becomes non-finite counts as 10% (chance) from then on.
* **Paired difference** for one seed: `d = A(config 1) - A(config 2)` on that seed. Reported as mean,
  standard deviation, paired standard error `sd / sqrt(n)`, and the number of seeds with `d > 0`.

## Main run: 8 configurations, 20 epochs, seeds 0 to 4, depth 2, lr 0.05

`fp32`, `dgrad`, `dgrad+sr`, `dgrad-dy`, `dgrad-dy+sr`, `dgrad-w`, `fprop+dgrad+wgrad`,
`fprop+dgrad+wgrad+sr`. (`dgrad-dy` rounds only the gradient dY in Dgrad, W stays FP32; `dgrad-w`
rounds only W, dY stays FP32.)

**H1. Round-to-nearest gradients hurt, and SR fixes them.** Supported if, for both pairs
(`dgrad+sr` vs `dgrad`) and (`fprop+dgrad+wgrad+sr` vs `fprop+dgrad+wgrad`):
mean `d` > 2 paired standard errors, and `d > 0` in at least 4 of 5 seeds.
**H1b. SR recovers FP32.** Supported if `|mean d|` < 1 point for `dgrad+sr` vs `fp32` and for
`fprop+dgrad+wgrad+sr` vs `fp32`.

**H2. It is the rounding of dY, not of W.** Supported if all three hold:
(a) `dgrad-dy` is below `fp32` by more than 1 point and by more than 2 paired standard errors;
(b) `|mean d|` < 1 point for `dgrad-w` vs `fp32`;
(c) `|mean d|` < 1 point for `dgrad-dy+sr` vs `fp32`.

**H3. Failure, not slower learning.** For `dgrad` vs `fp32`, compare the gap in test accuracy
(mean over seeds) at epoch 3 with the gap at epoch 20. "Genuine failure" if the gap at epoch 20 is at
least as large as at epoch 3; "slower learning" if it is less than half; "partly closes" in between.

## Sensitivity: does the conclusion depend on settings?

* **Learning rate:** lr 0.01 and 0.1 (0.05 comes from the main run), seeds 0 and 1, 10 epochs,
  configurations `fp32`, `dgrad`, `dgrad+sr`. Metric A over epochs 8 to 10.
* **Depth:** 1 and 4 hidden layers (2 comes from the main run), seeds 0 to 2, 10 epochs, the same
  three configurations.
* Reported as "holds" at a setting if mean `d` for `dgrad+sr` vs `dgrad` is positive and larger
  than 2 paired standard errors there, and "does not hold" otherwise. With 2 or 3 seeds these are
  checks of direction, not precise estimates.

## Gradient diagnostics

`apps/fp4_grad_error.py` on CIFAR-10 at steps 0, 391, 1173 (after 0, 1 and 3 epochs of FP32 training),
including `dgrad-dy`, `dgrad-dy+sr` and `dgrad-w`: relative error, cosine similarity, and for SR the
split into bias (error of the mean of 32 draws) and noise (spread of single draws around that mean).
