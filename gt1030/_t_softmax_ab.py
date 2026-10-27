# -*- coding: utf-8 -*-
"""A/B: 旧 softmax_causal / softmax_bwd_causal vs 向量化版本。"""
import sys
import time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM

d = MM.Dev()
d.activate()
ctx, q, mf = d.ctx, d.q, cl.mem_flags
T, HD, BH = MM.T, MM.HD, MM.BH
scale = 0.125
rng = np.random.default_rng(0)
S = (rng.standard_normal((BH, T, T)) * 0.7).astype(np.float16)
dS = cl.Buffer(ctx, mf.READ_WRITE, S.nbytes)
P = cl.Buffer(ctx, mf.READ_WRITE, S.nbytes)
P2 = cl.Buffer(ctx, mf.READ_WRITE, S.nbytes)
dP = (rng.standard_normal((BH, T, T)) * 0.02).astype(np.float16)
dPb = cl.Buffer(ctx, mf.READ_WRITE, dP.nbytes)
cl.enqueue_copy(q, dPb, dP)
Sb = cl.Buffer(ctx, mf.READ_WRITE, S.nbytes)
cl.enqueue_copy(q, Sb, S)


def t(fn, n=10):
    fn(); d.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    d.finish()
    return (time.time() - t0) / n * 1000


old = t(lambda: d.K['softmax_causal'](q, (T * 256, BH), (256, 1), Sb, P, np.int32(T),
                                      np.float32(scale)))
new = t(lambda: d.K['softmax_causal_v'](q, (T * 256, BH), (256, 1), Sb, P2, np.int32(T),
                                        np.float32(scale)))
a = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q, a, P)
b = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q, b, P2)
d.finish()
rel = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max() / (np.abs(a.astype(np.float32)).max() + 1e-12))
print('softmax fwd   old %6.3f -> new %6.3f ms  %.2fx   rel=%.2e' % (old, new, old / new, rel))

old_b = t(lambda: d.K['softmax_bwd_causal'](q, (T * 256, BH), (256, 1), P, dPb, dS,
                                            np.int32(T), np.float32(scale)))
new_b = t(lambda: d.K['softmax_bwd_causal_v'](q, (T * 256, BH), (256, 1), P, dPb, P2,
                                              np.int32(T), np.float32(scale)))
c = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q, c, dS)
e = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q, e, P2)
d.finish()
rel2 = float(np.abs(c.astype(np.float32) - e.astype(np.float32)).max() / (np.abs(c.astype(np.float32)).max() + 1e-12))
print('softmax bwd   old %6.3f -> new %6.3f ms  %.2fx   rel=%.2e' % (old_b, new_b, old_b / new_b, rel2))
print('每层省 %.2f ms ⇒ 10 层 %.0f ms/步' % (old + old_b - new - new_b, (old + old_b - new - new_b) * 10))
