# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
"""Experiment: how much error does each FP4 matmul put into the gradients?

Training curves are noisy, so this measures the effect of FP4 on the gradients
directly. For one fixed batch and one set of weights:

  1. compute exact FP32 weight gradients of every layer (the reference),
  2. compute them again with each QuantConfig (same weights, same batch),
  3. report, per layer, the relative error ||g_fp4 - g_fp32|| / ||g_fp32||,
     the cosine similarity, and the fraction of gradient entries that became
     exactly zero (FP4 underflow).

For stochastic-rounding configs it also averages the gradient over K random
draws. If SR is unbiased, the error of that average shrinks like 1/sqrt(K);
round-to-nearest is deterministic, so averaging cannot remove its error (bias).

It does this at several points of FP32 training (--checkpoints), because
gradients shrink as training progresses, which is when FP4 underflow bites.

    python3 apps/fp4_grad_error.py --device cpu --checkpoints 0 469 1407
    python3 apps/fp4_grad_error.py --dataset cifar10 --checkpoints 0 391 1173

NumPy is used only for the metrics, like the other apps.
"""
import argparse
import csv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.append(os.path.dirname(__file__))

import numpy as np

import needle as ndl
from needle.fp4.config import QuantConfig
from fp4_experiment import DATASETS, make_mlp, to_device

CONFIGS = [
    QuantConfig(fprop=True),
    QuantConfig(dgrad=True),
    QuantConfig(dgrad=True, stochastic=True),
    QuantConfig(wgrad=True),
    QuantConfig(wgrad=True, stochastic=True),
    QuantConfig(fprop=True, dgrad=True, wgrad=True),
    QuantConfig(fprop=True, dgrad=True, wgrad=True, stochastic=True),
]
LAYERS = ["layer1", "layer2", "layer3"]   # the three QuantizedLinear layers of make_mlp


def weight_grads(model, X, y):
    """Weight gradients of the three linear layers for one batch, as NumPy arrays."""
    for p in model.parameters():
        p.grad = None
    loss = ndl.nn.SoftmaxLoss()(model(X), y)
    loss.backward()
    linears = [m for m in model.modules if isinstance(m, ndl.nn.QuantizedLinear)]
    return [m.weight.grad.numpy().astype(np.float64) for m in linears]


def copy_weights(src, dst):
    for p_dst, p_src in zip(dst.parameters(), src.parameters()):
        p_dst.data = p_src.data


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["cpu", "cuda", "cpu_numpy"], default="cpu")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", choices=DATASETS, default="mnist")
    parser.add_argument("--out", default="results/grad_error.csv")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 469, 1407],
                        help="FP32 training steps after which gradients are measured")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sr-draws", type=int, default=32, help="K draws averaged for SR configs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = getattr(ndl, args.device)()
    if not device.enabled():
        parser.error(f"needle's {args.device} backend is not built. On Colab: Runtime > "
                     "Change runtime type > GPU, then rebuild with `make`.")
    load, in_features = DATASETS[args.dataset]
    train_set, test_set = load(args.data_dir, 0)
    np.random.seed(args.seed)
    model = make_mlp(QuantConfig(), args.hidden, device, in_features)   # FP32 model we train
    optimizer = ndl.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # One fixed evaluation batch from the test set, used at every checkpoint.
    X_eval, y_eval = to_device(next(iter(ndl.data.DataLoader(test_set, args.batch_size))), device)

    rows, step = [], 0
    batches = iter(())
    for checkpoint in sorted(args.checkpoints):
        # Train the FP32 model up to this checkpoint.
        while step < checkpoint:
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(ndl.data.DataLoader(train_set, args.batch_size, shuffle=True))
                batch = next(batches)
            X, y = to_device(batch, device)
            optimizer.reset_grad()
            ndl.nn.SoftmaxLoss()(model(X), y).backward()
            optimizer.step()
            step += 1

        reference = weight_grads(model, X_eval, y_eval)
        print(f"\nstep {step}: FP32 |dW| per layer = "
              + ", ".join(f"{np.linalg.norm(g):.3g}" for g in reference))
        print(f"{'config':24s} {'layer':7s} {'rel.err':>8s} {'cosine':>7s} {'zeros%':>7s} "
              f"{'rel.err of mean of K':>21s}")

        for config in CONFIGS:
            qmodel = make_mlp(config, args.hidden, device, in_features)
            copy_weights(model, qmodel)
            grads = weight_grads(qmodel, X_eval, y_eval)
            # For SR, average K independent draws to expose (lack of) bias.
            if config.stochastic:
                mean = [g.copy() for g in grads]
                for _ in range(args.sr_draws - 1):
                    for m, g in zip(mean, weight_grads(qmodel, X_eval, y_eval)):
                        m += g
                mean = [m / args.sr_draws for m in mean]
            for name, g, g_ref, i in zip(LAYERS, grads, reference, range(3)):
                ref_norm = np.linalg.norm(g_ref)
                rel_err = np.linalg.norm(g - g_ref) / ref_norm
                cosine = float(np.sum(g * g_ref) / (np.linalg.norm(g) * ref_norm + 1e-30))
                zeros = float(np.mean((g == 0) & (g_ref != 0))) * 100
                mean_err = (np.linalg.norm(mean[i] - g_ref) / ref_norm) if config.stochastic else rel_err
                rows.append({"step": step, "config": config.name, "layer": name,
                             "rel_error": f"{rel_err:.5f}", "cosine": f"{cosine:.5f}",
                             "zero_pct": f"{zeros:.2f}", "rel_error_of_mean": f"{mean_err:.5f}",
                             "sr_draws": args.sr_draws if config.stochastic else 1})
                print(f"{config.name:24s} {name:7s} {rel_err:8.4f} {cosine:7.4f} {zeros:7.2f} "
                      f"{mean_err:21.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
