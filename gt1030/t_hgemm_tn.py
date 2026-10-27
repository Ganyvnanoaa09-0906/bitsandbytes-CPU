# -*- coding: utf-8 -*-
"""Sanity: does hgemm_tn(A(M,K), B(M,N)) really equal A^T @ B ?"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.attention_gpu as A
A.B, A.T = 2, 1024
A.M = A.B * A.T
A.BH = A.B * A.NH
import gt1030.full_gpu_layer as L

ctx, q, mf = A.ctx, A.queue, A.mf
M, K, N = L.M, 512, 2736
rng = np.random.default_rng(0)
a = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
b = (rng.standard_normal((M, N)) * 0.1).astype(np.float16)
ab = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
bb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b)
c = cl.Buffer(ctx, mf.READ_WRITE, K * N * 2)
L.hgemm_tn(ab, bb, c, K, N, M)
q.finish()
out = np.empty((K, N), np.float16)
cl.enqueue_copy(q, out, c); q.finish()
ref = a.astype(np.float32).T @ b.astype(np.float32)
err = np.abs(out.astype(np.float32) - ref).max()
rel = err / (np.abs(ref).max() + 1e-12)
print('hgemm_tn %dx%dx%d: max_abs=%.4e rel=%.4e  outnorm=%.4f refnorm=%.4f'
      % (M, K, N, err, rel, np.abs(out.astype(np.float32)).max(), np.abs(ref).max()))

# also fp16-accumulation baseline for reference
ref16 = (a.astype(np.float32).T @ b.astype(np.float32)).astype(np.float16)
print('fp32-in/fp16-out baseline rel=%.4e' % (np.abs(ref16.astype(np.float32) - ref).max() / np.abs(ref).max()))
