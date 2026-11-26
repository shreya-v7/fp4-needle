# Extended experiments: verdicts

Criteria from `docs/PREREGISTRATION.md`, applied by `apps/fp4_stats.py`.

## Main run (20 epochs, metric A = mean test accuracy of the last 3 epochs)

| config | seeds | metric A (%) | final epoch (%) |
|---|---|---|---|
| fp32 | 5 | 52.28 ± 0.19 | 52.28 |
| dgrad | 5 | 47.76 ± 0.68 | 48.86 |
| dgrad+sr | 5 | 51.90 ± 0.27 | 51.98 |
| dgrad-dy | 5 | 47.80 ± 1.67 | 47.72 |
| dgrad-dy+sr | 5 | 51.96 ± 0.27 | 51.67 |
| dgrad-w | 5 | 52.09 ± 0.24 | 52.01 |
| fprop+dgrad+wgrad | 5 | 12.38 ± 1.90 | 13.08 |
| fprop+dgrad+wgrad+sr | 5 | 46.14 ± 1.97 | 46.63 |

### Pre-registered verdicts

* dgrad+sr vs dgrad: mean +4.14 points, sd 0.80, paired SE 0.36, positive in 5/5 seeds
* fprop+dgrad+wgrad+sr vs fprop+dgrad+wgrad: mean +33.75 points, sd 2.53, paired SE 1.13, positive in 5/5 seeds
* **H1 (round-to-nearest gradients hurt, SR fixes them): SUPPORTED**
* dgrad+sr vs fp32: mean -0.39 points, sd 0.18, paired SE 0.08, positive in 0/5 seeds
* fprop+dgrad+wgrad+sr vs fp32: mean -6.14 points, sd 1.99, paired SE 0.89, positive in 0/5 seeds
* **H1b (SR recovers FP32, within 1 point): NOT SUPPORTED**
* (a) dgrad-dy vs fp32: mean -4.48 points, sd 1.56, paired SE 0.70, positive in 0/5 seeds -> yes
* (b) dgrad-w vs fp32: mean -0.20 points, sd 0.18, paired SE 0.08, positive in 1/5 seeds -> yes
* (c) dgrad-dy+sr vs fp32: mean -0.32 points, sd 0.16, paired SE 0.07, positive in 0/5 seeds -> yes
* **H2 (it is the rounding of dY, not of W): SUPPORTED**
* **H3: FP32 minus dgrad is 9.15 points at epoch 3 and 3.42 at epoch 20: slower learning**

Gap to FP32 by epoch (mean over seeds, points):

| epoch | dgrad | dgrad+sr | dgrad-dy | dgrad-dy+sr | dgrad-w | fprop+dgrad+wgrad | fprop+dgrad+wgrad+sr |
|---|---|---|---|---|---|---|---|
| 1 | -2.0 | -0.4 | -2.7 | -0.3 | -0.0 | -11.3 | -0.6 |
| 2 | -5.0 | -0.4 | -3.6 | -0.4 | -0.3 | -18.9 | -0.8 |
| 3 | -9.1 | +0.1 | -6.7 | +0.2 | +0.3 | -21.2 | -0.7 |
| 4 | -4.1 | -0.8 | -3.4 | -0.3 | +0.1 | -24.7 | -1.5 |
| 5 | -3.9 | -0.8 | -4.2 | -0.5 | -0.4 | -26.1 | -1.9 |
| 6 | -5.3 | -0.5 | -4.1 | -0.2 | -0.0 | -26.9 | -2.0 |
| 7 | -3.7 | -0.4 | -3.4 | -0.5 | +0.2 | -26.2 | -1.5 |
| 8 | -6.7 | -0.4 | -3.9 | -0.3 | +0.1 | -30.2 | -3.2 |
| 9 | -6.1 | -0.5 | -6.5 | -0.5 | -0.2 | -32.8 | -3.9 |
| 10 | -4.8 | -0.2 | -4.8 | -0.7 | -0.4 | -33.5 | -3.3 |
| 11 | -3.4 | -0.3 | -4.1 | -0.3 | +0.2 | -35.7 | -3.6 |
| 12 | -3.4 | -0.7 | -5.0 | -0.6 | -0.1 | -37.9 | -4.3 |
| 13 | -3.5 | -0.9 | -4.0 | -0.7 | -0.8 | -37.0 | -5.8 |
| 14 | -4.4 | -0.9 | -4.5 | -0.6 | -0.1 | -40.5 | -5.1 |
| 15 | -4.3 | -0.6 | -4.8 | -0.6 | -0.1 | -39.2 | -5.6 |
| 16 | -2.0 | -0.5 | -2.9 | -0.1 | -0.4 | -34.8 | -6.3 |
| 17 | -2.8 | -0.6 | -3.8 | -1.1 | -0.3 | -40.8 | -6.8 |
| 18 | -5.3 | -0.3 | -4.9 | -0.2 | +0.1 | -40.9 | -6.3 |
| 19 | -4.9 | -0.6 | -4.0 | -0.2 | -0.4 | -39.6 | -6.5 |
| 20 | -3.4 | -0.3 | -4.6 | -0.6 | -0.3 | -39.2 | -5.7 |

## Learning-rate sweep (10 epochs, metric A over epochs 8 to 10)

| setting | fp32 (%) | dgrad (%) | dgrad+sr (%) | dgrad+sr minus dgrad | holds? |
|---|---|---|---|---|---|
| 0.01 | 45.81 | 34.32 | 44.94 | +10.62 (SE 0.76, 2/2 seeds) | yes |
| 0.05 | 49.29 | 43.42 | 48.91 | +5.49 (SE 0.23, 5/5 seeds) | yes |
| 0.1 | 49.63 | 46.95 | 48.71 | +1.76 (SE 0.35, 2/2 seeds) | yes |

## Depth sweep (hidden layers) (10 epochs, metric A over epochs 8 to 10)

| setting | fp32 (%) | dgrad (%) | dgrad+sr (%) | dgrad+sr minus dgrad | holds? |
|---|---|---|---|---|---|
| 1 | 48.15 | 42.96 | 47.51 | +4.55 (SE 2.05, 3/3 seeds) | yes |
| 2 | 49.29 | 43.42 | 48.91 | +5.49 (SE 0.23, 5/5 seeds) | yes |
| 4 | 49.09 | 43.64 | 49.07 | +5.43 (SE 0.45, 3/3 seeds) | yes |
