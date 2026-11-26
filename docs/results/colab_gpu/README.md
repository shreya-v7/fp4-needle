# Colab GPU results (first run)

From `notebooks/fp4_experiments.ipynb` on a Colab T4 (320 GB/s peak bandwidth).

- `bench_gemv_v1.*`: GEMV benchmark with the first NVFP4 kernel. Our FP32 GEMV reaches 84% of peak bandwidth;
  the first NVFP4 GEMV moved 7x fewer bytes but ran at 0.59x the FP32 speed at 4096 x 4096. The kernel was
  reworked after this run.
- `mnist/`: the MNIST grid and ablations (3 epochs, all data), seeds 1 and 2. They match the CPU runs in `cpu_sgd/`.
