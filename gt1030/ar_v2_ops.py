# -*- coding: utf-8 -*-
"""M1: iGPU kernels for V2 ops missing from the existing stack.
 - qk_rmsnorm_fwd: per-head RMSNorm over HD=64 (q and k), weights per head
 - scale_vec_fwd: elementwise out = x * s (broadcast s over rows)
Correctness harness vs torch fp32.
"""
from __future__ import annotations
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import torch

import gt1030.attention_gpu as A

B, T, MD, HD, NH = 2, 1024, 512, 64, 8

KSRC = r"""
__kernel void qk_rmsnorm_fwd(__global const half* x, __global const float* w,
                             __global half* y, int BH, int T, int HD) {
    int bh = get_global_id(0);
    int t  = get_global_id(1);
    int b = bh / (BH / 8);            // unused, placeholder kept simple
    __global const half* row = x + ((long)bh * T + t) * HD;
    __global half* out = y + ((long)bh * T + t) * HD;
    float ss = 0.f;
    for (int i = 0; i < HD; ++i) { float v = (float)row[i]; ss += v * v; }
    float inv = rsqrt(ss / HD + 1e-6f);
    for (int i = 0; i < HD; ++i)
        out[i] = (half)((float)row[i] * inv * w[i]);
}
__kernel void scale_vec_fwd(__global const half* x, __global const float* s,
                            __global half* y, int rows, int D) {
    int i = get_global_id(0);
    if (i >= rows * D) return;
    y[i] = (half)((float)x[i] * s[i % D]);
}
__kernel void scale_vec_bwd_dy(__global const half* dy, __global const float* s,
                               __global half* dyd, int rows, int D) {
    int i = get_global_id(0);
    if (i >= rows * D) return;
    dyd[i] = (half)((float)dy[i] * s[i % D]);
}
"""


def main():
    ctx, q, mf = A.ctx, A.queue, A.mf
    prog = cl.Program(ctx, KSRC).build(options=['-cl-mad-enable'])
    k_fwd = cl.Kernel(prog, 'qk_rmsnorm_fwd')
    k_scale = cl.Kernel(prog, 'scale_vec_fwd')

    rng = np.random.default_rng(0)
    BHn = B * NH
    x = (rng.standard_normal((BHn, T, HD)) * 0.5).astype(np.float16)
    w = (np.ones(HD, dtype=np.float32) * 1.0)
    y = np.empty_like(x)
    dx = np.empty_like(x)
    dw = np.zeros(HD, dtype=np.float32)

    xb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    wb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w)
    yb = cl.Buffer(ctx, mf.WRITE_ONLY, y.nbytes)
    k_fwd(q, (BHn, T), (1, 1), xb, wb, yb, np.int32(BHn), np.int32(T), np.int32(HD))
    cl.enqueue_copy(q, y, yb); q.finish()

    xt = torch.from_numpy(x.astype(np.float32)).view(BHn, T, HD)
    wt = torch.from_numpy(w)
    ref = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6) * wt
    err = float((torch.from_numpy(y.astype(np.float32)) - ref).abs().max())
    rel = err / float(ref.abs().max())
    print('qk_norm fwd max_abs=%.4e rel=%.4e %s' % (err, rel, 'OK' if rel < 1e-2 else 'FAIL'))

    s = (rng.standard_normal(MD).astype(np.float32) * 0.1 + 0.5)
    xv = (rng.standard_normal((B * T, MD)) * 0.5).astype(np.float16)
    yv = np.empty_like(xv)
    xvb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xv)
    sb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s)
    yvb = cl.Buffer(ctx, mf.WRITE_ONLY, yv.nbytes)
    k_scale(q, (B * T * MD,), None, xvb, sb, yvb, np.int32(B * T), np.int32(MD))
    cl.enqueue_copy(q, yv, yvb); q.finish()
    refv = torch.from_numpy(xv.astype(np.float32)) * torch.from_numpy(s)[None, :]
    errv = float((torch.from_numpy(yv.astype(np.float32)) - refv).abs().max())
    relv = errv / float(refv.abs().max())
    print('scale fwd max_abs=%.4e rel=%.4e %s' % (errv, relv, 'OK' if relv < 1e-2 else 'FAIL'))


if __name__ == '__main__':
    main()
