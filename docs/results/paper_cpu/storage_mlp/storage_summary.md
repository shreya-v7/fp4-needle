| format | bits | SQNR dB, rot none | SQNR dB, rot 16 | SQNR dB, rot 128 |
|---|---|---|---|---|
| nvfp4 | 4.500 | 20.21 | 20.40 | 20.44 |
| nvint4 | 4.500 | 22.13 | 21.23 | 21.31 |
| mxfp4 | 4.250 | 18.82 | 18.62 | 18.62 |
| int4_g32 | 4.500 | 21.24 | 20.19 | 20.21 |
| int4_g128 | 4.125 | 19.62 | 18.69 | 18.50 |
| nf4_g64 | 4.250 | 20.91 | 20.77 | 20.74 |

INT4 error model kappa^2/588: median measured/predicted = 1.00, log-log correlation = 0.994 over 6 tensors
NVINT4 beats NVFP4 (no rotation) on 100% of 3 tensors; median kappa^2 where INT wins 3.67, where FP wins nan
