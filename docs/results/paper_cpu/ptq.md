| checkpoint | metric | format | bits | no rotation | rotation 16 | rotation 128 |
|---|---|---|---|---|---|---|
| mlp_fp32_s0.npz | test_acc | fp32 | 32.000 | 0.9758 |  |  |
| mlp_fp32_s0.npz | test_acc | nvfp4 | 4.500 | 0.9755 | 0.9744 | 0.9754 |
| mlp_fp32_s0.npz | test_acc | nvint4 | 4.500 | 0.9744 | 0.9753 | 0.9758 |
| mlp_fp32_s0.npz | test_acc | mxfp4 | 4.250 | 0.9734 | 0.9759 | 0.9744 |
| mlp_fp32_s0.npz | test_acc | int4_g32 | 4.500 | 0.9756 | 0.9758 | 0.9749 |
| mlp_fp32_s0.npz | test_acc | int4_g128 | 4.125 | 0.9753 | 0.9766 | 0.9761 |
| mlp_fp32_s0.npz | test_acc | nf4_g64 | 4.250 | 0.9757 | 0.9748 | 0.9749 |
| lstm_fp32_s0.npz | test_ppl | fp32 | 32.000 | 666.9955 |  |  |
| lstm_fp32_s0.npz | test_ppl | nvfp4 | 4.500 | 671.0558 | 668.4655 | 669.1691 |
| lstm_fp32_s0.npz | test_ppl | nvint4 | 4.500 | 667.9560 | 668.6244 | 668.0982 |
| lstm_fp32_s0.npz | test_ppl | mxfp4 | 4.250 | 669.5655 | 670.3037 | 659.3694 |
| lstm_fp32_s0.npz | test_ppl | int4_g32 | 4.500 | 669.7431 | 667.9795 | 671.1033 |
| lstm_fp32_s0.npz | test_ppl | int4_g128 | 4.125 | 669.5939 | 669.8299 | 669.0778 |
| lstm_fp32_s0.npz | test_ppl | nf4_g64 | 4.250 | 668.9116 | 669.5594 | 669.3582 |
