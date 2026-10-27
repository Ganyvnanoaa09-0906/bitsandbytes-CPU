# -*- coding: utf-8 -*-
"""量一下层间 x 传输（down 2MB + up 2MB + finish）真实成本，再决定要不要做 device→device 链。"""
import sys
import time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM

d = MM.Dev()
d.activate()
ctx, q = d.ctx, d.q
M, MD = MM.M, MM.MD
x = (np.random.randn(M, MD) * 0.3).astype(np.float16)
xb = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, x.nbytes)
xb2 = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, x.nbytes)
cl.enqueue_copy(q, xb, x)
d.finish()


def t(fn, n=30):
    fn(); d.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    d.finish()
    return (time.time() - t0) / n * 1000


print('down 2MB (device->host, 需同步) : %.3f ms' % t(lambda: d.down(xb, (M, MD), np.float16)))
print('up   2MB (host->device)        : %.3f ms' % t(lambda: cl.enqueue_copy(q, xb, x)))
print('dev->dev 2MB                   : %.3f ms' % t(lambda: cl.enqueue_copy(q, xb2, xb)))
print('down+up 一对（现在每层两次）    : %.3f ms'
      % t(lambda: (d.down(xb, (M, MD), np.float16), cl.enqueue_copy(q, xb, x))))
