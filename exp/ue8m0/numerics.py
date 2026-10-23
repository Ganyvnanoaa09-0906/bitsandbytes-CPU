#!/usr/bin/env python3
"""Numerics study: fp32-absmax blockwise quantization (bnb's current scheme) vs
ue8m0 (power-of-two shared exponent) blockwise quantization.

Motivation
----------
DeepInfer / DeepGEMM / DeepSeek's MX formats use per-block scales that are powers of two
(shared exponent, ue8m0). On CPU this is interesting because a power-of-two scale makes the
dequantization an exact exponent shift, which in turn allows an *integer* dot product:
    y = sum_k (qx_k * 2^sx) * (qw_k * 2^sw)  =  2^(sx+sw) * sum_k (qx_k * qw_k)
i.e. the per-element float multiply disappears and the inner loop becomes int8 x int8 -> int32
(`vpmaddubsw` + `vpmaddwd` on AVX2), with one float multiply per block instead of per element.

This script answers the numerical half of the question: how much error does the power-of-two
scale add, for (a) weights only (bnb style: activations stay fp32) and (b) both weight and
activations quantized (the integer-dot-product regime)?

Metrics are relative ones, so they are comparable across magnitudes.
"""

import itertools
import json
import math
import sys

import numpy as np

RNG = np.random.default_rng(20261023)


# ---------------------------------------------------------------- distributions
def make_tensor(kind: str, shape, rng: np.random.Generator) -> np.ndarray:
    """Realistic weight/state distributions with a consistent std of ~1 before scaling."""
    n = int(np.prod(shape))
    if kind == "gauss":
        return rng.standard_normal(shape).astype(np.float32) * 0.02
    if kind == "laplace":
        return (rng.laplace(0.0, 1.0, shape) * 0.02).astype(np.float32)
    if kind == "uniform":
        return rng.uniform(-1, 1, shape).astype(np.float32) * 0.02
    if kind == "outliers":  # 0.1% entries x20, as seen in real weight matrices
        x = rng.standard_normal(shape).astype(np.float32) * 0.02
        idx = rng.choice(n, size=max(1, n // 1000), replace=False)
        x.flat[idx] *= 20.0
        return x
    if kind == "gauss_wide":  # heavier tails than gauss, common for activations
        return (rng.standard_t(5, shape) * 0.02).astype(np.float32)
    raise ValueError(kind)


# ---------------------------------------------------------------- schemes
def blocks_along_last(x: np.ndarray, blocksize: int):
    """Yield (row, block_index, start, end) for contiguous blocks along the last axis."""
    n = x.shape[-1]
    nblocks = (n + blocksize - 1) // blocksize
    for b in range(nblocks):
        s = b * blocksize
        e = min(s + blocksize, n)
        yield b, s, e


def quant_absmax_8bit(x: np.ndarray, blocksize: int):
    """bnb-style: per-block scale = max|x| / 127, int8 codes in [-127, 127]."""
    out = np.empty_like(x)
    scales = np.empty(x.shape[:-1] + ((x.shape[-1] + blocksize - 1) // blocksize,), np.float32)
    flat = x.reshape(-1, x.shape[-1])
    outf = out.reshape(-1, x.shape[-1])
    scf = scales.reshape(-1, scales.shape[-1])
    for r in range(flat.shape[0]):
        row = flat[r]
        for b, s, e in blocks_along_last(row, blocksize):
            am = float(np.max(np.abs(row[s:e])))
            k = am / 127.0 if am > 0 else 1.0
            scf[r, b] = k
            outf[r, s:e] = np.clip(np.rint(row[s:e] / k), -127, 127)
    return outf.astype(np.int8), scf


def quant_ue8m0_8bit(x: np.ndarray, blocksize: int):
    """ue8m0-style: per-block scale = 2^ceil(log2(max|x| / 127)), int8 codes."""
    out = np.empty_like(x)
    scales = np.empty(x.shape[:-1] + ((x.shape[-1] + blocksize - 1) // blocksize,), np.float32)
    flat = x.reshape(-1, x.shape[-1])
    outf = out.reshape(-1, x.shape[-1])
    scf = scales.reshape(-1, scales.shape[-1])
    for r in range(flat.shape[0]):
        row = flat[r]
        for b, s, e in blocks_along_last(row, blocksize):
            am = float(np.max(np.abs(row[s:e])))
            if am == 0.0:
                k = 1.0
            else:
                # 2^ceil(log2(am/127)) >= am/127  =>  codes fit in [-127,127]
                k = float(2.0 ** math.ceil(math.log2(am / 127.0)))
            scf[r, b] = k
            outf[r, s:e] = np.clip(np.rint(row[s:e] / k), -127, 127)
    return outf.astype(np.int8), scf


def dequant(codes: np.ndarray, scales: np.ndarray, blocksize: int) -> np.ndarray:
    out = np.empty(codes.shape, np.float32)
    flat = codes.reshape(-1, codes.shape[-1]).astype(np.float32)
    scf = scales.reshape(-1, scales.shape[-1])
    outf = out.reshape(-1, codes.shape[-1])
    for r in range(flat.shape[0]):
        for b, s, e in blocks_along_last(flat, blocksize):
            outf[r, s:e] = flat[r, s:e] * scf[r, b]
    return outf


# ---------------------------------------------------------------- metrics
def rel_error(ref: np.ndarray, test: np.ndarray) -> float:
    denom = np.linalg.norm(ref.astype(np.float64))
    if denom == 0:
        return 0.0
    return float(np.linalg.norm((ref - test).astype(np.float64)) / denom)


def run_case(kind: str, blocksize: int, shape, act_shape=None):
    w = make_tensor(kind, shape, RNG)

    # ---- A: bnb current scheme (fp32 absmax scale), weights only
    cw_a, sw_a = quant_absmax_8bit(w, blocksize)
    w_a = dequant(cw_a, sw_a, blocksize)

    # ---- B: ue8m0 scale, weights only
    cw_b, sw_b = quant_ue8m0_8bit(w, blocksize)
    w_b = dequant(cw_b, sw_b, blocksize)

    # ---- C: ue8m0 scale for BOTH weights and activations (integer dot regime)
    if act_shape is None:
        act_shape = (64, shape[-1])
    x = make_tensor("gauss_wide", act_shape, RNG)
    cx, sx = quant_ue8m0_8bit(x, blocksize)
    xq = dequant(cx, sx, blocksize)
    ca_a, sa_a = quant_absmax_8bit(x, blocksize)
    xa = dequant(ca_a, sa_a, blocksize)

    res = {
        "dist": kind,
        "blocksize": blocksize,
        "shape": list(shape),
        "weight_err_absmax": rel_error(w, w_a),
        "weight_err_ue8m0": rel_error(w, w_b),
        "act_err_absmax": rel_error(x, xa),
        "act_err_ue8m0": rel_error(x, xq),
    }
    # End-to-end GEMM: y = x @ W^T, comparing quantization schemes against fp32 reference
    y_ref = x @ w.T
    y_a = x @ w_a.T          # A: float activations, quantized weights (bnb)
    y_b = x @ w_b.T          # B: same but pow2 scale
    y_c = xq @ w_b.T         # C: both quantized (integer dot regime)
    res["gemm_err_absmax"] = rel_error(y_ref, y_a)
    res["gemm_err_ue8m0"] = rel_error(y_ref, y_b)
    res["gemm_err_int8_both"] = rel_error(y_ref, y_c)
    return res


def main():
    kinds = ["gauss", "laplace", "uniform", "outliers", "gauss_wide"]
    blocksizes = [64, 128, 256, 512, 4096]
    rows = []
    for kind, bs in itertools.product(kinds, blocksizes):
        rows.append(run_case(kind, bs, (256, 2048)))
        sys.stderr.write(f"  done {kind} bs={bs}\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
