# Paper experiments: CPU results

Produced locally on CPU with `apps/fp4_storage.py` and `apps/plot_paper.py`.

- `storage_gpt2/`: all 48 matmul weights of GPT-2 small. SQNR per format and rotation, the INT4 error model fit, NVINT4 vs NVFP4 win rate.
- `storage_mlp/`, `storage_lstm/`: the same for our FP32-trained MNIST MLP and PTB LSTM, plus `ptq.csv` (test metric after quantizing the weights).
- `ptq.md`: post-training quantization table for both models (columns: no rotation, rotation 16, rotation 128).

GPU results (sqrt(3) check, GEMV benchmark, CUDA tests) come from `notebooks/fp4_experiments.ipynb`.
- `threshold_mlp/`: the sqrt(3) check on the MNIST MLP (CPU, seed 0, 2 epochs, ratio logged every 25 steps): FP32, all-FP4, all-FP4 + SR, and all-FP4 switched to an FP32 backward pass at step 469 (end of epoch 1).
