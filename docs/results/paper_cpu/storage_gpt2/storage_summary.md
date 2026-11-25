| format | bits | SQNR dB, rot none | SQNR dB, rot 16 | SQNR dB, rot 128 |
|---|---|---|---|---|
| nvfp4 | 4.500 | 20.47 | 20.42 | 20.43 |
| nvint4 | 4.500 | 21.08 | 21.33 | 21.29 |
| mxfp4 | 4.250 | 18.66 | 18.76 | 18.76 |
| int4_g32 | 4.500 | 19.96 | 20.30 | 20.28 |
| int4_g128 | 4.125 | 17.93 | 18.51 | 18.65 |
| nf4_g64 | 4.250 | 20.56 | 20.72 | 20.73 |

INT4 error model kappa^2/588: median measured/predicted = 0.97, log-log correlation = 0.969 over 96 tensors
NVINT4 beats NVFP4 (no rotation) on 98% of 48 tensors; median kappa^2 where INT wins 4.66, where FP wins 6.22
