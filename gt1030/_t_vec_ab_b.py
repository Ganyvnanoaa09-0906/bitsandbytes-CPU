# -*- coding: utf-8 -*-
"""Stage B 的 A/B：旧 split/merge/qk_norm vs 新向量化版本（同进程背靠背，不受系统负载漂移影响）。"""
import sys
import time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM

d = MM.Dev()
d.activate()
ctx, q, mf = d.ctx, d.q, cl.mem_flags
T, HD, NH, NKV, B, M, WROW = MM.T, MM.HD, MM.NH, MM.NKV, MM.B, MM.M, MM.WROW
BH = MM.BH
rng = np.random.default_rng(0)


def up(a):
    b = cl.Buffer(ctx, mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(q, b, a)
    return b


def t(fn, n=20):
    fn(); d.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    d.finish()
    return (time.time() - t0) / n * 1000


qkv = up((rng.standard_normal((M, WROW)) * 0.2).astype(np.float16))
Qraw = cl.Buffer(ctx, mf.READ_WRITE, BH * T * HD * 2)
Kraw = cl.Buffer(ctx, mf.READ_WRITE, BH * T * HD * 2)
Vb = cl.Buffer(ctx, mf.READ_WRITE, BH * T * HD * 2)
o_tok = cl.Buffer(ctx, mf.READ_WRITE, M * HD * NH * 2)
qw = up((rng.standard_normal(HD) * 0.1 + 1).astype(np.float32))
dQ = up((rng.standard_normal((BH * T, HD)) * 0.05).astype(np.float16))
dQraw = cl.Buffer(ctx, mf.READ_WRITE, BH * T * HD * 2)
dqkv = cl.Buffer(ctx, mf.READ_WRITE, M * WROW * 2)
dKraw = up((rng.standard_normal((BH * T, HD)) * 0.05).astype(np.float16))
dV = up((rng.standard_normal((BH * T, HD)) * 0.05).astype(np.float16))

rows = []


def ab(tag, old, new):
    o, nn = t(old), t(new)
    rows.append((tag, o, nn))
    print('  %-22s old %7.3f -> new %7.3f ms   %.2fx' % (tag, o, nn, (o / nn) if nn > 1e-9 else 0))


ab('split_q',
   lambda: d.K['split_q'](q, (BH * T * HD,), (256,), qkv, Qraw, np.int32(T), np.int32(HD),
                           np.int32(NH), np.int32(NKV)),
   lambda: d.K['split_q_v'](q, (BH * T * HD // 8,), (128,), qkv, Qraw, np.int32(T),
                            np.int32(HD), np.int32(NH), np.int32(NKV)))
ab('split_kv',
   lambda: d.K['split_kv'](q, (BH * T * HD,), (256,), qkv, Kraw, Vb, np.int32(T), np.int32(HD),
                           np.int32(NH), np.int32(NKV), np.int32(B)),
   lambda: d.K['split_kv_v'](q, (BH * T * HD // 8,), (128,), qkv, Kraw, Vb, np.int32(T),
                             np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B)))
ab('merge_o',
   lambda: d.K['merge_o'](q, (BH * T * HD,), (256,), Qraw, o_tok, np.int32(T), np.int32(HD),
                          np.int32(NH), np.int32(B)),
   lambda: d.K['merge_o_v'](q, (BH * T * HD // 8,), (128,), Qraw, o_tok, np.int32(T),
                            np.int32(HD), np.int32(NH), np.int32(B)))
ab('split_dout',
   lambda: d.K['split_dout'](q, (BH * T * HD,), (256,), o_tok, Qraw, np.int32(T), np.int32(HD),
                             np.int32(NH), np.int32(B)),
   lambda: d.K['split_dout_v'](q, (BH * T * HD // 8,), (128,), o_tok, Qraw, np.int32(T),
                               np.int32(HD), np.int32(NH), np.int32(B)))
ab('qk_norm_fwd',
   lambda: d.K['qk_norm_fwd'](q, (BH * T,), None, Qraw, qw, dQraw, np.int32(BH * T), np.int32(HD)),
   lambda: d.K['qk_norm_fwd_v'](q, (BH * T * 8,), (8,), Qraw, qw, dQraw, np.int32(BH * T),
                                np.int32(HD)))
ab('qk_norm_dx',
   lambda: d.K['qk_norm_dx'](q, (BH * T,), None, Qraw, qw, dQ, dQraw, np.int32(BH * T), np.int32(HD)),
   lambda: d.K['qk_norm_dx_v'](q, (BH * T * 8,), (8,), Qraw, qw, dQ, dQraw, np.int32(BH * T),
                               np.int32(HD)))
ab('merge_dqkv',
   lambda: d.K['merge_dqkv'](q, (M * WROW,), (256,), dQraw, dKraw, dV, dqkv, np.int32(T),
                             np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B)),
   lambda: (d.K['merge_dqkv_v'](q, (BH * T * HD // 8,), (128,), dQraw, dKraw, dV, dqkv,
                                np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B)),
            d.K['merge_dkv_v'](q, (B * 2 * NKV * T * HD // 8,), (128,), dKraw, dV, dqkv,
                               np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))))

old_t = sum(r[1] for r in rows); new_t = sum(r[2] for r in rows)
print('\n小计：old %.1f ms -> new %.1f ms (%.2fx)，每层省 %.1f ms ⇒ 10 层 %.0f ms/步'
      % (old_t, new_t, old_t / new_t, old_t - new_t, (old_t - new_t) * 10))
