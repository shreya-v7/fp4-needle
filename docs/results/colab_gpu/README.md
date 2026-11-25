# Colab GPU results (first run)

From `notebooks/fp4_experiments.ipynb` on a Colab GPU with 320 GB/s peak bandwidth.

- `bench_gemv_v1.*`: GEMV benchmark with the first NVFP4 kernel. Our FP32 GEMV reaches 84% of peak bandwidth;
  the NVFP4 GEMV moves 7x fewer bytes but is slower (0.59x of FP32 at 4096 x 4096), so it was limited by
  instructions and latency, not memory. The kernel was reworked after this run (float4 loads of x, shared-memory
  E4M3 table, two blocks in flight); a new benchmark replaces this one once it has run on a GPU.
- `mnist/`: the MNIST grid and ablations, full settings (3 epochs, all data), seeds 1 and 2. Seed 0 of this
  run came from a QUICK run and is left out.
- `storage/`, `threshold/`: format study and sqrt(3) check from the QUICK run (1 epoch, 10,000 images).
