# -*- coding: utf-8 -*-
"""逐个验证 Stage A 的新 kernel 与 numpy 参考是否一致。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM
import gt1030.attention_gpu as A

ctx, mf = A.ctx, cl.mem_flags
K = mm = MM
d = MM.Dev()
d.activate()
ctx = d.ctx          # Dev.activate() 会把 A.ctx/A.queue 换成自己的 context
q = d.q
mf = cl.mem_flags

MD, M, FF = MM.MD, MM.M, MM.FF
EPS = 1e-6
rng = np.random.default_rng(0)


def up(a, dtype=None):
    b = cl.Buffer(ctx, mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(d.q, b, a)
    return b


def down(b, shape, dtype=np.float16):
    a = np.empty(shape, dtype)
    cl.enqueue_copy(d.q, a, b)
    d.finish()
    return a


# ---- 1) rms_fwd_fused ----
x = (rng.standard_normal((M, MD)) * 0.3).astype(np.float16)
w = (rng.standard_normal(MD) * 0.5 + 1.0).astype(np.float16)
xb, wb = up(x), up(w)
yb = cl.Buffer(ctx, mf.READ_WRITE, x.nbytes)
ssb = cl.Buffer(ctx, mf.READ_WRITE, M * 4)
d.K['rms_fwd_fused'](d.q, (M * 64,), (64,), xb, wb, yb, ssb, np.int32(M), np.int32(MD))
d.finish()
y = down(yb, (M, MD)).astype(np.float32)
ss = down(ssb, M, np.float32)
xf = x.astype(np.float32)
ref_ss = (xf ** 2).mean(axis=1)
ref_y = xf * (1.0 / np.sqrt(ref_ss[:, None] + EPS)) * w.astype(np.float32)[None, :]
print('rms_fwd   y rel=%.2e   ss rel=%.2e'
      % (np.abs(y - ref_y).max() / np.abs(ref_y).max(),
         np.abs(ss - ref_ss).max() / np.abs(ref_ss).max()))

# ---- 2) rms_bwd_fused ----
dh = (rng.standard_normal((M, MD)) * 0.05).astype(np.float16)
dhb = up(dh)
dxb = cl.Buffer(ctx, mf.READ_WRITE, x.nbytes)
ss2 = cl.Buffer(ctx, mf.READ_WRITE, M * 4)
sc2 = cl.Buffer(ctx, mf.READ_WRITE, M * 4)
d.K['rms_bwd_fused'](d.q, (M * 64,), (64,), xb, dhb, wb, dxb, ss2, sc2, np.int32(M), np.int32(MD))
d.finish()
dx = down(dxb, (M, MD)).astype(np.float32)
ssv = down(ss2, M, np.float32)
scv = down(sc2, M, np.float32)
hv = dh.astype(np.float32)
wv = w.astype(np.float32)
sig = np.sqrt(ref_ss + EPS)
s_ref = ((hv * wv[None, :]) * xf).sum(axis=1)
coef = s_ref / (MD * sig ** 3)
ref_dx = (hv * wv[None, :]) / sig[:, None] - xf * coef[:, None]
print('rms_bwd   dx rel=%.2e   ss rel=%.2e   sc rel=%.2e'
      % (np.abs(dx - ref_dx).max() / np.abs(ref_dx).max(),
         np.abs(ssv - ref_ss).max() / np.abs(ref_ss).max(),
         np.abs(scv - s_ref).max() / np.abs(s_ref).max()))

# ---- 3) scale_add_v ----
s = (rng.standard_normal(MD)).astype(np.float32)
yy = (rng.standard_normal((M, MD)) * 0.2).astype(np.float16)
sb = up(s)
yyb = up(yy)
ob = cl.Buffer(ctx, mf.READ_WRITE, yy.nbytes)
d.K['scale_add_v'](d.q, (M * MD // 8,), (128,), xb, sb, yyb, ob, np.int32(M), np.int32(MD))
d.finish()
o = down(ob, (M, MD)).astype(np.float32)
ref_o = xf + s[None, :] * yy.astype(np.float32)
print('scale_add rel=%.2e' % (np.abs(o - ref_o).max() / np.abs(ref_o).max()))

# ---- 4) scale_v ----
d.K['scale_v'](d.q, (M * MD // 8,), (128,), xb, sb, ob, np.int32(M), np.int32(MD))
d.finish()
o = down(ob, (M, MD)).astype(np.float32)
print('scale     rel=%.2e' % (np.abs(o - xf * s[None, :]).max() / np.abs(xf * s[None, :]).max()))

# ---- 5) split_silu_v / bwd ----
a2 = (rng.standard_normal((M, 2 * FF)) * 1.2).astype(np.float16)
a2b = up(a2)
gb = cl.Buffer(ctx, mf.READ_WRITE, M * FF * 2)
d.K['split_silu_v'](d.q, (M * FF // 8,), (128,), a2b, gb, np.int32(M), np.int32(FF))
d.finish()
g = down(gb, (M, FF)).astype(np.float32)
af = a2.astype(np.float32)
ga, gb2 = af[:, :FF], af[:, FF:]
ref_g = ga / (1 + np.exp(-ga)) * gb2
print('split_silu   rel=%.2e' % (np.abs(g - ref_g).max() / np.abs(ref_g).max()))

dg = (rng.standard_normal((M, FF)) * 0.05).astype(np.float16)
dgb = up(dg)
da2b = cl.Buffer(ctx, mf.READ_WRITE, a2.nbytes)
d.K['split_silu_bwd_v'](d.q, (M * FF // 8,), (128,), a2b, dgb, da2b, np.int32(M), np.int32(FF))
d.finish()
da2 = down(da2b, (M, 2 * FF)).astype(np.float32)
sgi = 1 / (1 + np.exp(-ga))
dsilu = sgi * (1 + ga * (1 - sgi))
ref_da = np.concatenate([dg.astype(np.float32) * gb2 * dsilu,
                         dg.astype(np.float32) * ga * sgi], axis=1)
print('split_silu_b rel=%.2e' % (np.abs(da2 - ref_da).max() / np.abs(ref_da).max()))
