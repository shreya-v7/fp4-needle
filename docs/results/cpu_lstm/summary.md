| config | seeds | final test acc (%) | final test loss | final test ppl | train loss, last 100 steps | s per run |
|---|---|---|---|---|---|---|
| fp32 | 3 | 11.17 ± 0.32 | 6.4994 | 664.7 ± 1.7 | 5.9845 | 217 |
| fprop | 3 | 11.11 ± 0.20 | 6.5090 | 671.2 ± 3.7 | 5.9933 | 257 |
| fprop+dgrad | 3 | 11.12 ± 0.27 | 6.5138 | 674.4 ± 3.2 | 6.0023 | 308 |
| fprop+dgrad+wgrad | 3 | 11.09 ± 0.01 | 6.4790 | 651.3 ± 1.6 | 6.0019 | 333 |
| fprop+dgrad+wgrad+sr | 3 | 11.09 ± 0.25 | 6.5124 | 673.4 ± 2.6 | 5.9989 | 344 |
