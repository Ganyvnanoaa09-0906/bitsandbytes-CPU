# -*- coding: utf-8 -*-
"""探针 39：对齐到 64KB 的 arena sub-buffer，400 个混合尺寸的写读一致性。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM

ALIGN = 64 * 1024
d = MM.Dev()
q, ctx, mf = d.q, d.ctx, d.mf
MB = 1 << 20
ARENA = 1500 * MB
arena = cl.Buffer(ctx, mf.READ_WRITE, ARENA)
info = arena.get_info(cl.mem_info.BASE_ADDR_ALIGN if hasattr(cl.mem_info, 'BASE_ADDR_ALIGN') else 0x1129)
print('BASE_ADDR_ALIGN = %d' % info)

rng = np.random.default_rng(0)
sizes = [int(x * MB) for x in (0.5, 1, 1.4, 2.8, 5.6, 2, 4, 8)]
off = ALIGN
subs = []
while off < ARENA - 16 * MB and len(subs) < 400:
    nb = sizes[len(subs) % len(sizes)]
    o = (off + ALIGN - 1) // ALIGN * ALIGN
    if o + nb > ARENA:
        break
    subs.append((arena.get_sub_region(o, nb), nb, float((len(subs) % 50) + 1)))
    off = o + nb
print('created %d sub-buffers, %.1f MB' % (len(subs), (off - ALIGN) / 1e6))
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
print('不一致 %d / %d' % (bad, len(subs)))
