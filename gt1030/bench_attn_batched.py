# -*- coding: utf-8 -*-
"""bench_attn_batched.py — 测 attention 反向用的 CLBlast StridedBatched hgemm 达成率。

形状（AR 30M, T=1024, BH=16）：
  dV = P^T @ dO : m=T, n=HD, k=T   (1024 x 128 x 1024), 16 batch
  dK = dS^T @ Q : m=T, n=HD, k=T   (1024 x 128 x 1024)
  QK^T          : m=T, n=T,  k=HD  (1024 x 1024 x 128), 16 batch

用法：python gt1030/bench_attn_batched.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402

A = MM.A
T, HD, BH = int(MM.T), int(MM.HD), int(MM.BH)


def bench(name, m, n, k, ta_tr, tb_tr):
    mf = cl.mem_flags
    a = cl.Buffer(A.ctx, mf.READ_ONLY, BH * m * k * 2)
    b = cl.Buffer(A.ctx, mf.READ_ONLY, BH * k * n * 2)
    c = cl.Buffer(A.ctx, mf.READ_WRITE, BH * m * n * 2)
    def call():
        A.hgemm_batched(a, b, c, m, n, k, ta_tr, tb_tr,
                        k if ta_tr == A.NO else m,
                        n if tb_tr == A.NO else k,
                        n, m * k, k * n, m * n)
    call(); A.queue.finish()
    best = 1e9
    for _ in range(10):
        t0 = time.perf_counter(); call(); A.queue.finish()
        best = min(best, time.perf_counter() - t0)
    flops = 2.0 * BH * m * n * k
    g = flops / best / 1e9
    print(f"  {name:14s} BH={BH} {m}x{n}x{k}  {best*1e3:7.2f} ms  {g:7.1f} GFLOPS  "
          f"({100*g/2300:4.1f}% peak)  per-step total={10*g and flops*10/1e9:5.1f} GFLOP", flush=True)
    return g, flops


def main() -> int:
    print(f"  iGPU fp16 peak 2300 GFLOPS | T={T} BH={BH} HD={HD}")
    r1 = bench("dV=P^T@dO", T, HD, T, A.YES, A.NO)
    r2 = bench("dK=dS^T@Q", T, HD, T, A.YES, A.NO)
    r3 = bench("QK^T", T, T, HD, A.NO, A.YES)
    # 10 层、fwd+bwd 的 attention GEMM 总时间估算
    tot_s = 0.0
    for g, fl in (r1, r2, r3):
        tot_s += 10 * fl / g / 1e9 if g > 0 else 0.0
    print(f"  估计 10 层 attention GEMM 合计 ≈ {tot_s:.3f} s/步（fwd QK/AV 用自定义 kernel，未计）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
