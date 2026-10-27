# -*- coding: utf-8 -*-
"""测：大 buffer 的『小量回读』是否不可靠（8 字节 vs 全量）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

d = MM.Dev()
q, ctx, mf = d.q, d.ctx, d.mf
rng = np.random.default_rng(0)
for shape in [(512, 512), (512, 1024), (512, 2736), (1368, 512), (2048, 512)]:
    a = (rng.standard_normal(shape) * 0.03).astype(np.float16)
    b = cl.Buffer(ctx, mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(q, b, np.ascontiguousarray(a), is_blocking=True)
    q.finish()
    chk8 = np.empty(8, np.uint8)
    cl.enqueue_copy(q, chk8, b)
    q.finish()
    want8 = np.ascontiguousarray(a).view(np.uint8)[:8]
    full = np.empty(a.size, np.float16)
    cl.enqueue_copy(q, full, b)
    q.finish()
    print('%-12s bytes=%7d  8B_read_ok=%-5s  full_read_ok=%-5s  full_absmax=%.4f'
          % (shape, a.nbytes, np.array_equal(chk8, want8),
             np.allclose(full, a.ravel(), atol=0), float(np.abs(full).max())))
