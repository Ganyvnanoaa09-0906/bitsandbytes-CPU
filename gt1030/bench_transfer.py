"""How fast can data actually move between host and the iGPU on this machine?

The naive executor measured ~1 GB/s effective for pageable numpy buffers, which is what makes its
per-GEMM overhead explode. This measures the alternatives at the sizes the AR step uses (2-8 MB):
  a) direct enqueue_copy from/to a pageable numpy array
  b) the same copy where the host side is a USE_HOST_PTR cl.Buffer (driver may pin/register it)
  c) device-to-device copy (upper bound of what a resident-activation design would pay)
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
print('device:', ctx.devices[0].name)


def timeit(fn, iters=20):
    fn()
    queue.finish()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        queue.finish()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


print('%-8s %14s %14s %14s %14s' % ('size', 'H2D direct', 'D2H direct', 'H2D pinned', 'D2H pinned'))
for mb in (0.5, 1, 2, 4, 8, 16):
    n = int(mb * 1024 * 1024)
    host = np.zeros(n, dtype=np.uint8)
    dev = cl.Buffer(ctx, mf.READ_WRITE, n)
    h2d = timeit(lambda: cl.enqueue_copy(queue, dev, host))
    d2h = timeit(lambda: cl.enqueue_copy(queue, host, dev))
    # "pinned": host memory registered as a cl.Buffer, then device<->device copy
    hbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=host)
    hbuf2 = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=host)
    a = timeit(lambda: cl.enqueue_copy(queue, dev, hbuf))
    b = timeit(lambda: cl.enqueue_copy(queue, hbuf2, dev))
    print('%-8s %9.2f GB/s %9.2f GB/s %9.2f GB/s %9.2f GB/s'
          % ('%dMB' % mb, mb / 1024 / h2d if h2d else 0, mb / 1024 / d2h if d2h else 0,
             mb / 1024 / a if a else 0, mb / 1024 / b if b else 0))
