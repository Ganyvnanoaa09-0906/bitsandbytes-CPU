# -*- coding: utf-8 -*-
"""Probe 35: how many MIXED-SIZE buffers can coexist in one context before reads go bad?"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM

d = MM.Dev()
q, ctx, mf = d.q, d.ctx, d.mf
MB = 1 << 20
sizes = [int(0.5 * MB), 1 * MB, int(1.4 * MB), int(2.8 * MB), int(5.6 * MB), 2 * MB, 4 * MB,
         int(33.5 * MB)]
sent = cl.Buffer(ctx, mf.READ_WRITE, 4096)
pat = np.full(2048, 0.75, np.float16)
cl.enqueue_copy(q, sent, pat, is_blocking=True)
q.finish()
bufs = []
bad = None
for i in range(300):
    nb = sizes[i % len(sizes)]
    b = cl.Buffer(ctx, mf.READ_WRITE, nb)
    bufs.append(b)
    v = np.full(nb // 2, float((i % 100) + 1), np.float16)
    cl.enqueue_copy(q, b, v, is_blocking=True)
    q.finish()
    out = np.empty(2048, np.float16)
    cl.enqueue_copy(q, out, sent)
    q.finish()
    o2 = np.empty(8, np.float16)
    cl.enqueue_copy(q, o2, b)
    q.finish()
    if (not np.array_equal(out, pat) or float(o2[0]) != float((i % 100) + 1)) and bad is None:
        bad = i
        print('*** breaks at buffer #%d (size %.1fMB), total %.1f MB, sentinel_ok=%s val_ok=%s'
              % (i, nb / 1e6, sum(int(x.size) for x in bufs) / 1e6,
                 np.array_equal(out, pat), float(o2[0])))
        break
if bad is None:
    print('all %d mixed-size buffers OK, %.1f MB' % (len(bufs),
                                                     sum(int(x.size) for x in bufs) / 1e6))
