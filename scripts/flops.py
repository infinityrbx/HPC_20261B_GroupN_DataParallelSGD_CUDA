#!/usr/bin/env python3
"""FLOP count per training step for the MLP in src/cuda/train_dp.cu, split into
matrix multiplication and everything else, plus one conv layer done as im2col + matmul.

  flops.py [--hidden 512 ...] [--batch 256] [--step-ms 0.855]

--step-ms (optional): measured fwd+bwd time per step, to print achieved GFLOP/s.
"""
import argparse

IN, C = 784, 10  # src/common/config.h: CFG_INPUT, CFG_CLASSES

def mlp(H, B):
    P = IN * H + H + H * C + C
    mm = {  # the five gpu_matmul calls in train_dp.cu, 2*M*N*K each
        "fwd  Z1 = X . W1":      2 * B * H * IN,
        "fwd  Z2 = A1 . W2":     2 * B * C * H,
        "bwd  dW2 = A1^T . dZ2": 2 * H * C * B,
        "bwd  dA1 = dZ2 . W2^T": 2 * B * H * C,
        "bwd  dW1 = X^T . dA1":  2 * IN * H * B,
    }
    other = {  # element-wise / row-wise kernels, ~ops per element
        "bias + relu":          2 * B * H + 1 * B * C,
        "softmax + xent grad":  5 * B * C,
        "relu backward":        1 * B * H,
        "bias grads (colsum)":  B * H + B * C,
        "sgd update":           2 * P,
    }
    return mm, other, P

def conv_im2col(Hh=28, Ww=28, Cin=32, Cout=64, k=3):
    M, K, N = Hh * Ww, k * k * Cin, Cout
    return M, K, N, 2 * M * K * N

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, nargs="+", default=[512])
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--step-ms", type=float)
a = ap.parse_args()

for H in a.hidden:
    mm, other, P = mlp(H, a.batch)
    smm, soth = sum(mm.values()), sum(other.values())
    print(f"\nMLP 784-{H}-10, batch {a.batch}, {P:,} params")
    for k, v in mm.items():
        print(f"  {k:<24}{v/1e6:10.2f} MFLOP")
    for k, v in other.items():
        print(f"  {k:<24}{v/1e6:10.2f} MFLOP")
    print(f"  matmul {smm/1e6:.1f} MFLOP, other {soth/1e6:.2f} MFLOP -> "
          f"matmul = {100*smm/(smm+soth):.2f}% of FLOPs")
    print(f"  per sample {smm/a.batch/1e6:.2f} MFLOP; per epoch (60k) {smm/a.batch*60000/1e9:.1f} GFLOP")
    if a.step_ms and len(a.hidden) == 1:
        print(f"  achieved {smm/(a.step_ms*1e-3)/1e9:.0f} GFLOP/s at {a.step_ms} ms/step "
              f"(T4 FP32 peak ~8100 GFLOP/s)")

M, K, N, f = conv_im2col()
print(f"\nConv 3x3, 32->64 channels, 28x28 output, as im2col + matmul:")
print(f"  im2col matrix {M} x {K}, weights {K} x {N} -> one ({M}x{K})({K}x{N}) matmul = {f/1e6:.1f} MFLOP per image")
