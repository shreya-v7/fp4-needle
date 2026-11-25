| format | bits | SQNR dB, rot none | SQNR dB, rot 16 | SQNR dB, rot 128 |
|---|---|---|---|---|
| nvfp4 | 4.500 | 20.12 | 20.45 | 20.47 |
| nvint4 | 4.500 | 22.15 | 21.22 | 21.26 |
| mxfp4 | 4.250 | 19.03 | 18.73 | 18.70 |
| int4_g32 | 4.500 | 21.46 | 20.25 | 20.18 |
| int4_g128 | 4.125 | 20.81 | 19.38 | 19.31 |
| nf4_g64 | 4.250 | 20.75 | 20.70 | 20.71 |

INT4 error model kappa^2/588: median measured/predicted = 0.98, log-log correlation = 0.993 over 6 tensors
NVINT4 beats NVFP4 (no rotation) on 100% of 3 tensors; median kappa^2 where INT wins 3.70, where FP wins nan
