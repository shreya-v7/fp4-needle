| checkpoint | metric | format | bits | no rotation | rotation 16 | rotation 128 |
|---|---|---|---|---|---|---|
| mlp_fp32_s0.npz | test_acc | fp32 | 32.000 | 0.8499 |  |  |
| mlp_fp32_s0.npz | test_acc | nvfp4 | 4.500 | 0.8475 | 0.8409 | 0.8518 |
| mlp_fp32_s0.npz | test_acc | nvint4 | 4.500 | 0.8447 | 0.8482 | 0.8474 |
| mlp_fp32_s0.npz | test_acc | mxfp4 | 4.250 | 0.8444 | 0.8494 | 0.8501 |
| mlp_fp32_s0.npz | test_acc | int4_g32 | 4.500 | 0.8459 | 0.8497 | 0.8443 |
| mlp_fp32_s0.npz | test_acc | int4_g128 | 4.125 | 0.8497 | 0.8452 | 0.8439 |
| mlp_fp32_s0.npz | test_acc | nf4_g64 | 4.250 | 0.8331 | 0.8450 | 0.8464 |
| lstm_fp32_s0.npz | test_ppl | fp32 | 32.000 | 761.9694 |  |  |
| lstm_fp32_s0.npz | test_ppl | nvfp4 | 4.500 | 763.2756 | 762.3944 | 766.2796 |
| lstm_fp32_s0.npz | test_ppl | nvint4 | 4.500 | 765.9290 | 763.5458 | 762.7667 |
| lstm_fp32_s0.npz | test_ppl | mxfp4 | 4.250 | 765.3796 | 764.0082 | 760.3404 |
| lstm_fp32_s0.npz | test_ppl | int4_g32 | 4.500 | 763.7367 | 766.8501 | 764.2567 |
| lstm_fp32_s0.npz | test_ppl | int4_g128 | 4.125 | 764.1769 | 762.1314 | 767.2883 |
| lstm_fp32_s0.npz | test_ppl | nf4_g64 | 4.250 | 762.9398 | 764.4324 | 763.3460 |
