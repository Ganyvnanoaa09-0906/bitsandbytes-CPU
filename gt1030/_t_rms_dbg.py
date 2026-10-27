# -*- coding: utf-8 -*-
"""定位 rms_fwd_fused 的数值错误：打印实际 vs 参考。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM

d = MM.Dev()
d.activate()
ctx, q, mf = d.ctx, d.q, cl.mem_flags
MD = MM.MD
EPS = 1e-6
M = 8
rng = np.random.default_rng(0)
x = (rng.standard_normal((M, MD)) * 0.3).astype(np.float16)
w = (rng.standard_normal(MD) * 0.5 + 1.0).astype(np.float16)


def up(a):
    b = cl.Buffer(ctx, mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(q, b, a)
    return b


xb, wb = up(x), up(w)
yb = cl.Buffer(ctx, mf.READ_WRITE, x.nbytes)
ssb = cl.Buffer(ctx, mf.READ_WRITE, M * 4)
d.K['rms_fwd_fused'](q, (M,), (64,), xb, wb, yb, ssb, np.int32(M), np.int32(MD))
d.finish()
y = np.empty((M, MD), np.float16); cl.enqueue_copy(q, y, yb)
ss = np.empty(M, np.float32); cl.enqueue_copy(q, ss, ssb)
d.finish()
xf = x.astype(np.float32)
ref_ss = (xf ** 2).mean(axis=1)
print('ss      ', np.array2string(ss[:4], precision=6))
print('ref_ss  ', np.array2string(ref_ss[:4], precision=6))
print('ratio   ', np.array2string(ss[:4] / ref_ss[:4], precision=4))
print('sum x^2 ', np.array2string((xf[:2] ** 2).sum(axis=1), precision=2))
print('y[0,:6] ', np.array2string(y[0, :6].astype(np.float32), precision=4))
ref_y = xf * (1.0 / np.sqrt(ref_ss[:, None] + EPS)) * w.astype(np.float32)[None, :]
print('ref_y   ', np.array2string(ref_y[0, :6], precision=4))
print('y last8 ', np.array2string(y[0, -8:].astype(np.float32), precision=4))
print('ref last', np.array2string(ref_y[0, -8:], precision=4))
