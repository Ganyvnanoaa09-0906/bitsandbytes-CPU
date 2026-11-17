# -*- coding: utf-8 -*-
"""bench_gemm_shapes.py — 在 AR 模型真实形状上测核显 GEMM 达成率（hgemm vs sgemm）。

目的：判断"核显端到端 0.33 s/步（追上 1030 理论）"的瓶颈是否可解。
  · fp16 峰值 2.30 TFLOPS、fp32 峰值 1.15 TFLOPS；1030 fp32 理论 1.128 TFLOPS
  · 若 hgemm 只能到 ~350 GFLOPS，则 0.33 s 不可能靠 CLBlast 达成（需要自定义 kernel）

用法：
  python gt1030/bench_gemm_shapes.py --reps 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402

import gt1030.full_gpu_step_v2 as V2  # noqa: E402
import gt1030.ar_v2_multi as MM  # noqa: E402

# (名字, M, N, K, 转置方式) —— 与 AR 30M 模型的真实一层前向/反向一致
M = int(MM.M)
L = MM.L
SHAPES = [
    ("qkv_fwd", M, 1024, 512, "nt"),
    ("out_fwd", M, 512, 512, "nt"),
    ("w12_fwd", M, 2736, 512, "nt"),
    ("w3_fwd", M, 512, 1368, "nt"),
    ("head_fwd", M, 1024, 512, "nt"),
    ("dO_tok_bwd", M, 1024, 512, "tn"),
    ("dw_qkv_bwd", 1024, 512, M, "tn"),
    ("dw_w12_bwd", 2736, 512, M, "tn"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    import pyopencl as cl
    ctx = MM.A.ctx
    q = MM.A.queue
    mf = cl.mem_flags
    bl = V2.bl

    rows = []
    for name, m, n, k, kind in SHAPES:
        A = cl.Buffer(ctx, mf.READ_ONLY, m * k * 2)
        B = cl.Buffer(ctx, mf.READ_ONLY, n * k * 2)
        C = cl.Buffer(ctx, mf.READ_WRITE, m * n * 2)
        fn = {"nn": L.hgemm_nn, "nt": L.hgemm_nt, "tn": L.hgemm_tn}[kind]
        def call():
            fn(A, B, C, m, n, k)
        call()
        q.finish()
        best = 1e9
        for _ in range(a.reps):
            t0 = time.perf_counter()
            call()
            q.finish()
            best = min(best, time.perf_counter() - t0)
        flops = 2.0 * m * n * k
        gflops = flops / best / 1e9
        rows.append({"shape": name, "m": m, "n": n, "k": k, "ms": round(best * 1e3, 3),
                     "gflops": round(gflops, 1),
                     "pct_fp16_peak": round(100 * gflops / 2300, 1)})
        print(f"  {name:12s} {m}x{n}x{k}  {best*1e3:7.2f} ms  {gflops:7.1f} GFLOPS  "
              f"({100*gflops/2300:4.1f}% of fp16 peak)", flush=True)

    mean = sum(r["gflops"] for r in rows) / len(rows)
    print("  " + json.dumps({"mean_gflops": round(mean, 1),
                             "implied_step_s_fwd_bwd": round(374.8 / mean, 3)},
                            ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
