# -*- coding: utf-8 -*-
"""A/B 对比：旧 kernel（1 元素/线程）vs 新 kernel（8 元素/线程 + 融合），同一个进程里直接比。
旧的 rms_stats/rms_apply/rms_bwd_*/scale_vec/add_vec/split_silu 都还在 program 里，可以直接调。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM

d = MM.Dev()
d.activate()
ctx, q, mf = d.ctx, d.q, cl.mem_flags
M, MD, FF = MM.M, MM.MD, MM.FF
rng = np.random.default_rng(0)
x = (rng.standard_normal((M, MD)) * 0.3).astype(np.float16)
w = (rng.standard_normal(MD) * 0.2 + 1.0).astype(np.float16)
dh = (rng.standard_normal((M, MD)) * 0.05).astype(np.float16)
a2 = (rng.standard_normal((M, 2 * FF)) * 1.2).astype(np.float16)
s = (rng.standard_normal(MD)).astype(np.float32)


def up(a):
    b = cl.Buffer(ctx, mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(q, b, a)
    return b


xb, wb, dhb, a2b, sb = up(x), up(w), up(dh), up(a2), up(s)
yb = cl.Buffer(ctx, mf.READ_WRITE, x.nbytes)
ssb = cl.Buffer(ctx, mf.READ_WRITE, M * 4)
scb = cl.Buffer(ctx, mf.READ_WRITE, M * 4)
gb = cl.Buffer(ctx, mf.READ_WRITE, M * FF * 2)
dgb = cl.Buffer(ctx, mf.READ_WRITE, M * FF * 2)
da2b = cl.Buffer(ctx, mf.READ_WRITE, a2.nbytes)


def t(fn, n=20):
    fn(); d.finish()
    import time
    t0 = time.time()
    for _ in range(n):
        fn()
    d.finish()
    return (time.time() - t0) / n * 1000


rows = []


def ab(tag, old, new):
    o = t(old); nn = t(new)
    rows.append((tag, o, nn))
    r = (o / nn) if nn > 1e-9 else 0.0
    print('  %-26s old %7.3f ms -> new %7.3f ms  %.2fx' % (tag, o, nn, r))


ab('rms 前向 (stats+apply)',
   lambda: (d.K['rms_stats'](q, (M,), (64,), xb, ssb, np.int32(M), np.int32(MD)),
            d.K['rms_apply'](q, (M * MD,), (256,), xb, wb, ssb, yb, np.int32(M), np.int32(MD))),
   lambda: d.K['rms_fwd_fused'](q, (M * 64,), (64,), xb, wb, yb, ssb, np.int32(M), np.int32(MD)))

ab('rms 反向 (stats+apply)',
   lambda: (d.K['rms_bwd_stats'](q, (M,), (64,), xb, dhb, wb, ssb, scb, np.int32(M), np.int32(MD)),
            d.K['rms_bwd_apply'](q, (M * MD,), (256,), xb, dhb, wb, ssb, scb, yb, np.int32(M), np.int32(MD))),
   lambda: d.K['rms_bwd_fused'](q, (M * 64,), (64,), xb, dhb, wb, yb, ssb, scb, np.int32(M), np.int32(MD)))

ab('scale+add (ls 残差)',
   lambda: (d.K['scale_vec'](q, (M * MD,), (256,), xb, sb, yb, np.int32(M), np.int32(MD)),
            d.K['add_vec'](q, (M * MD,), (256,), xb, yb, yb, np.int32(M * MD))),
   lambda: d.K['scale_add_v'](q, (M * MD // 8,), (128,), xb, sb, yb, yb, np.int32(M), np.int32(MD)))

ab('split_silu 前向',
   lambda: d.K['split_silu'](q, (M * FF,), (256,), a2b, gb, np.int32(M), np.int32(FF)),
   lambda: d.K['split_silu_v'](q, (M * FF // 8,), (128,), a2b, gb, np.int32(M), np.int32(FF)))

ab('split_silu 反向',
   lambda: d.K['split_silu_bwd'](q, (M * FF,), (256,), a2b, gb, da2b, np.int32(M), np.int32(FF)),
   lambda: d.K['split_silu_bwd_v'](q, (M * FF // 8,), (128,), a2b, gb, da2b, np.int32(M), np.int32(FF)))

ab('scale 单个',
   lambda: d.K['scale_vec'](q, (M * MD,), (256,), xb, sb, yb, np.int32(M), np.int32(MD)),
   lambda: d.K['scale_v'](q, (M * MD // 8,), (128,), xb, sb, yb, np.int32(M), np.int32(MD)))

ab('add 单个',
   lambda: d.K['add_vec'](q, (M * MD,), (256,), xb, xb, yb, np.int32(M * MD)),
   lambda: d.K['add_v'](q, (M * MD // 8,), (128,), xb, xb, yb, np.int32(M * MD)))

tot_old = sum(r[1] for r in rows)
tot_new = sum(r[2] for r in rows)
print('\n以上 7 组小计：旧 %.1f ms → 新 %.1f ms（%.2fx），每层省 %.1f ms ⇒ 10 层省 %.0f ms/步'
      % (tot_old, tot_new, tot_old / tot_new, tot_old - tot_new, (tot_old - tot_new) * 10))
