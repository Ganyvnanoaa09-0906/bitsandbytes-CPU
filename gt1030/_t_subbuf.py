# -*- coding: utf-8 -*-
"""探针 37：arena + clCreateSubBuffer（pyopencl 的 get_sub_region）能不能当普通 Buffer 用？
   如果行得通，就不用改任何 kernel/GEMM，只改分配方式。"""
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
print('arena %.2f GB' % (ARENA / 1e9))

# 1) 先测能不能切 sub-buffer
try:
    sub = arena.get_sub_region(mf.READ_WRITE, 1024, 2048)
    print('get_sub_region ->', type(sub), 'int_ptr=%#x size=%d offset=%d'
          % (sub.int_ptr, sub.size, getattr(sub, 'offset', -1)))
except Exception as e:
    print('get_sub_region 不可用: %r' % (e,))
    raise SystemExit(1)

# 2) 写读测试
v = np.arange(1024, dtype=np.float16)
cl.enqueue_copy(q, sub, v, is_blocking=True)
q.finish()
out = np.empty(1024, np.float16)
cl.enqueue_copy(q, out, sub)
q.finish()
print('sub write/read equal=%s' % np.array_equal(out, v))

# 3) 用 kernel 在 sub-buffer 上算 add_vec
K = d.K
a = np.full(4096, 0.5, np.float16)
b = np.full(4096, 0.25, np.float16)
sa = arena.get_sub_region(mf.READ_WRITE, 1 * MB, 4096 * 2)
sb = arena.get_sub_region(mf.READ_WRITE, 1 * MB + 8192, 4096 * 2)
sc = arena.get_sub_region(mf.READ_WRITE, 1 * MB + 16384, 4096 * 2)
cl.enqueue_copy(q, sa, a, is_blocking=True)
cl.enqueue_copy(q, sb, b, is_blocking=True)
q.finish()
K['add_vec'](q, (4096,), (256,), sa, sb, sc, np.int32(4096))
q.finish()
o = np.empty(4096, np.float16)
cl.enqueue_copy(q, o, sc)
q.finish()
print('kernel on sub-buffers: equal=%s (期望 0.75)  o[0]=%.3f' % (np.allclose(o, 0.75), o[0]))

# 4) 大量 sub-buffer（400 个，混合尺寸）写读一致性
rng = np.random.default_rng(0)
sizes = [int(x * MB) for x in (0.5, 1, 1.4, 2.8, 5.6, 2, 4, 8)]
subs = []
off = 64 * MB
refs = []
for i in range(400):
    nb = sizes[i % len(sizes)]
    if off + nb > ARENA:
        break
    s = arena.get_sub_region(mf.READ_WRITE, off, nb)
    off += nb
    subs.append(s)
    val = float((i % 50) + 1)
    vv = np.full(nb // 2, val, np.float16)
    cl.enqueue_copy(q, s, vv, is_blocking=True)
    q.finish()
    refs.append((s, nb, val))
bad = 0
for s, nb, val in refs:
    oo = np.empty(min(nb // 2, 4096), np.float16)
    cl.enqueue_copy(q, oo, s)
    q.finish()
    if float(oo[0]) != val:
        bad += 1
print('sub-buffers: %d 个, %.1f MB, 读回不一致 %d' % (len(subs), off / 1e6, bad))
