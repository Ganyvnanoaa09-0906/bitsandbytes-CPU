# -*- coding: utf-8 -*-
"""探针 38：正确的 get_sub_region 签名 (origin, size, flags) 测试。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM

d = MM.Dev()
q, ctx, mf = d.q, d.ctx, d.mf
MB = 1 << 20
ARENA = 1500 * MB
arena = cl.Buffer(ctx, mf.READ_WRITE, ARENA)
sub = arena.get_sub_region(1024, 2048)
print('sub ok: size=%d offset=%d int_ptr=%#x' % (sub.size, sub.offset, sub.int_ptr))
v = np.arange(1024, dtype=np.float16)
cl.enqueue_copy(q, sub, v, is_blocking=True)
q.finish()
out = np.empty(1024, np.float16)
cl.enqueue_copy(q, out, sub)
q.finish()
print('write/read equal=%s' % np.array_equal(out, v))

K = d.K
a = np.full(4096, 0.5, np.float16); b = np.full(4096, 0.25, np.float16)
sa = arena.get_sub_region(1 * MB, 8192)
sb = arena.get_sub_region(1 * MB + 8192, 8192)
sc = arena.get_sub_region(1 * MB + 16384, 8192)
cl.enqueue_copy(q, sa, a, is_blocking=True)
cl.enqueue_copy(q, sb, b, is_blocking=True)
q.finish()
K['add_vec'](q, (4096,), (256,), sa, sb, sc, np.int32(4096))
q.finish()
o = np.empty(4096, np.float16)
cl.enqueue_copy(q, o, sc)
q.finish()
print('kernel on subs: ok=%s o[0]=%.3f' % (np.allclose(o, 0.75), o[0]))

rng = np.random.default_rng(0)
sizes = [int(x * MB) for x in (0.5, 1, 1.4, 2.8, 5.6, 2, 4, 8)]
off, subs = 64 * MB, []
for i in range(400):
    nb = sizes[i % len(sizes)]
    if off + nb > ARENA:
        break
    s = arena.get_sub_region(off, nb)
    off += nb
    subs.append((s, nb, float((i % 50) + 1)))
for s, nb, val in subs:
    cl.enqueue_copy(q, s, np.full(nb // 2, val, np.float16), is_blocking=True)
q.finish()
bad = 0
for s, nb, val in subs:
    oo = np.empty(min(nb // 2, 4096), np.float16)
    cl.enqueue_copy(q, oo, s)
    q.finish()
    if float(oo[0]) != val:
        bad += 1
print('subs=%d  %.1f MB  不一致=%d' % (len(subs), off / 1e6, bad))
