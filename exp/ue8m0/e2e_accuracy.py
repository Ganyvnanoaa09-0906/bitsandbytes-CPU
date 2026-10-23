#!/usr/bin/env python3
"""End-to-end accuracy cost of quantizing activations for an integer-dot GEMM.

The per-matmul relative error in numerics.py is not enough to decide anything: what matters is
whether the model's *decisions* change. This script runs a small transformer-ish MLP block
(x -> W1 -> GELU -> W2 -> classifier head) on random inputs under three numeric regimes:

  fp32          reference
  w8-only       weights blockwise u8-quantized (what bnb's CPU gemm_8bit does today),
                activations stay fp32
  w8a8-int8     weights and activations blockwise int8 (what the integer-dot kernel needs)

and reports logit error, argmax (top-1) agreement and probability drift.
"""

import sys

import numpy as np

RNG = np.random.default_rng(20261023)


def blocks(x, bk):
    n = x.shape[-1]
    for s in range(0, n, bk):
        yield s, min(s + bk, n)


def q_w8_absmax(w, bk):
    """bnb-style: unsigned 8-bit linear code, fp32 absmax per block.

    `s` stores the dequant multiplier per code unit (absmax/127), NOT absmax itself --
    mixing the two conventions double-divides by 127 and inflates the output 127x.
    """
    q = np.empty_like(w)
    s = np.empty(w.shape[:-1] + ((w.shape[-1] + bk - 1) // bk,), np.float32)
    for r in range(w.shape[0]):
        row = w[r]
        for b, (lo, hi) in enumerate(blocks(row, bk)):
            am = float(np.max(np.abs(row[lo:hi])))
            step = am / 127.0 if am > 0 else 1.0
            s[r, b] = step
            q[r, lo:hi] = np.clip(np.rint(row[lo:hi] / step), -127, 127)
    return q, s


def q_s8_absmax(x, bk):
    """signed int8 with a per-block fp32 step (dequant multiplier per code unit)."""
    q = np.empty_like(x)
    s = np.empty(x.shape[:-1] + ((x.shape[-1] + bk - 1) // bk,), np.float32)
    for r in range(x.shape[0]):
        row = x[r]
        for b, (lo, hi) in enumerate(blocks(row, bk)):
            am = float(np.max(np.abs(row[lo:hi])))
            step = am / 127.0 if am > 0 else 1.0
            s[r, b] = step
            q[r, lo:hi] = np.clip(np.rint(row[lo:hi] / step), -127, 127)
    return q, s


def dq(q, s, bk):
    out = np.empty(q.shape, np.float32)
    for r in range(q.shape[0]):
        for b, (lo, hi) in enumerate(blocks(q[r], bk)):
            out[r, lo:hi] = q[r, lo:hi] * s[r, b]
    return out


def model_forward(x, W1, W2, W3):
    h = x @ W1.T
    h = h * (1.0 / (1.0 + np.exp(-1.7 * h)))  # gelu-ish, cheap
    h = h @ W2.T
    h = np.maximum(h, 0)
    return h @ W3.T


def run(bk, n_samples=200, d_in=512, d_h=2048, d_cls=10, scale=1.0):
    W1 = (RNG.standard_normal((d_h, d_in)) * 0.02).astype(np.float32)
    W2 = (RNG.standard_normal((d_in, d_h)) * 0.02).astype(np.float32)
    W3 = (RNG.standard_normal((d_cls, d_in)) * 0.02).astype(np.float32)
    x = (RNG.standard_normal((n_samples, d_in)) * scale).astype(np.float32)

    ref = model_forward(x, W1, W2, W3)

    # --- weights only (bnb today)
    q1, s1 = q_w8_absmax(W1, bk)
    q2, s2 = q_w8_absmax(W2, bk)
    q3, s3 = q_w8_absmax(W3, bk)
    w8 = model_forward(x, dq(q1, s1, bk), dq(q2, s2, bk), dq(q3, s3, bk))

    # --- weights + activations int8 (integer dot product regime)
    W1i, W1s = q_s8_absmax(W1, bk)
    W2i, W2s = q_s8_absmax(W2, bk)
    W3i, W3s = q_s8_absmax(W3, bk)

    def fwd_i8(xx):
        xi, xs = q_s8_absmax(xx, bk)
        xh = dq(xi, xs, bk)
        h = xh @ dq(W1i, W1s, bk).T
        h = h * (1.0 / (1.0 + np.exp(-1.7 * h)))
        hi, hs = q_s8_absmax(h, bk)
        h = dq(hi, hs, bk) @ dq(W2i, W2s, bk).T
        h = np.maximum(h, 0)
        hi2, hs2 = q_s8_absmax(h, bk)
        return dq(hi2, hs2, bk) @ dq(W3i, W3s, bk).T

    i8 = fwd_i8(x)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(a))

    agree_w8 = float(np.mean(np.argmax(ref, axis=1) == np.argmax(w8, axis=1)))
    agree_i8 = float(np.mean(np.argmax(ref, axis=1) == np.argmax(i8, axis=1)))
    # how much does the top-1 probability move
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    p_ref, p_w8, p_i8 = softmax(ref), softmax(w8), softmax(i8)
    return {
        "bk": bk,
        "logit_relerr_w8": rel(ref, w8),
        "logit_relerr_i8": rel(ref, i8),
        "top1_agree_w8": agree_w8,
        "top1_agree_i8": agree_i8,
        "max_prob_drift_w8": float(np.max(np.abs(p_ref - p_w8))),
        "max_prob_drift_i8": float(np.max(np.abs(p_ref - p_i8))),
    }


def main():
    print(f"{'BK':>6} | {'w8 logit err':>12} {'w8a8 logit err':>14} | "
          f"{'w8 top1':>8} {'w8a8 top1':>10} | {'w8 pdrift':>10} {'w8a8 pdrift':>12}")
    print("-" * 92)
    for bk in (64, 128, 256, 512, 1024):
        r = run(bk)
        print(f"{r['bk']:>6} | {r['logit_relerr_w8']:>12.3e} {r['logit_relerr_i8']:>14.3e} | "
              f"{r['top1_agree_w8']:>7.1%} {r['top1_agree_i8']:>9.1%} | "
              f"{r['max_prob_drift_w8']:>10.3e} {r['max_prob_drift_i8']:>12.3e}")


if __name__ == "__main__":
    main()
