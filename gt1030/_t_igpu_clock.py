# -*- coding: utf-8 -*-
"""探针 51：现在（CPU trainer 满载）核显还能跑多少 GFLOPS？
   —— 用来判断之前 4.0s/step 的测量是不是被 CPU 训练"饿"出来的。
   对照：报告 §7.110/§A.14 在 CPU 空闲时测到 fp16 GEMM 1039 GFLOPS、真实形状 371~655。"""
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM
import gt1030.attention_gpu as A
import gt1030.full_gpu_layer as L

ctx, q = A.ctx, A.queue
mf = cl.mem_flags
rng = np.random.default_rng(0)
cases = [('大 GEMM 2048x4096x4096', 2048, 4096, 4096),
         ('大 GEMM 4096x4096x4096', 4096, 4096, 4096),
         ('本模型 W12 2048x2736x512', 2048, 2736, 512),
         ('本模型 Wqkv 2048x1024x512', 2048, 1024, 512)]
for nm, M, N, K in cases:
    a = (rng.standard_normal((M, K)) * 0.05).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    cb = cl.Buffer(ctx, mf.READ_WRITE, M * N * 2)

    def run():
        L.hgemm_nn(ab, bb, cb, M, N, K)

    run(); q.finish()
    t0 = time.time()
    n = 5
    for _ in range(n):
        run()
    q.finish()
    dt = (time.time() - t0) / n
    print('%-26s %7.2f ms  %7.1f GFLOPS' % (nm, dt * 1000, 2.0 * M * N * K / dt / 1e9))
