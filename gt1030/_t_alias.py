# -*- coding: utf-8 -*-
"""探针 27：给若干不同 buffer 写不同图案，交叉回读，确认是否真的内存重叠。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

CKPT = r'F:\weibo_cos\ckpt_1l.pt'
dev = MM.V2ModelDev(CKPT)
d = dev.dev
q = d.q
M, MD = MM.M, MM.MD

targets = [
    ('tick', d._tick_buf, 1024, np.float16),
    ('L0.xin', dev.layers[0].b['xin'], M * MD, np.float16),
    ('L0.h1', dev.layers[0].b['h1'], M * MD, np.float16),
    ('logits', dev.sc['logits'], M * dev.V, np.float16),
    ('xa', dev.xa, M * MD, np.float16),
]
vals = {}
for i, (nm, buf, n, dt) in enumerate(targets):
    v = np.full(n, float(100 * (i + 1)), dt)
    cl.enqueue_copy(q, buf, v, is_blocking=True)
    q.finish()
    vals[nm] = float(100 * (i + 1))
for nm, buf, n, dt in targets:
    out = np.empty(n, dt)
    cl.enqueue_copy(q, out, buf)
    q.finish()
    got = float(out[0])
    print('%-8s 期望 %6.1f  读到 %8.1f  %s'
          % (nm, vals[nm], got, 'OK' if got == vals[nm] else '<<< 串了'))
