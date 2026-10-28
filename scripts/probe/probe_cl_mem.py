"""Probe why the resident engine dies when many 200 MB weight buffers are created.

Hypothesis: cl.Buffer(..., COPY_HOST_PTR, 202MB) x N exhausts/destabilises the driver (pinned staging),
even though the device reports 6.5 GB. Test three allocation strategies and then run a trivial kernel.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import sys

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ctx = cl.create_some_context(interactive=False)
q = cl.CommandQueue(ctx)
mf = cl.mem_flags
print('device:', ctx.devices[0].name,
      'global %.2f GB' % (ctx.devices[0].global_mem_size / 1e9),
      'max alloc %.2f GB' % (ctx.devices[0].max_mem_alloc_size / 1e9))

N = 202 * 1024 * 1024   # one layer's fp32 weights
host = np.zeros(N // 4, dtype=np.float32)


def trivial_ok():
    a = np.arange(256, dtype=np.float32)
    d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
    y = np.empty(256, dtype=np.float32)
    cl.enqueue_copy(q, y, d)
    q.finish()
    return float(np.abs(y - a).max()) < 1e-6


for strategy in ('copy_host_ptr', 'write_only+copy', 'rw+copy'):
    bufs = []
    limit = None
    for i in range(40):
        try:
            if strategy == 'copy_host_ptr':
                bufs.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host))
            elif strategy == 'write_only+copy':
                b = cl.Buffer(ctx, mf.READ_WRITE, N)
                cl.enqueue_copy(q, b, host)
                bufs.append(b)
            else:
                b = cl.Buffer(ctx, mf.READ_WRITE, N)
                cl.enqueue_copy(q, b, host)
                q.finish()
                bufs.append(b)
        except Exception as e:
            limit = (i, type(e).__name__, str(e)[:100])
            break
    q.finish()
    print('%-16s allocated %2d x 202MB (%.1f GB) before failure: %s | trivial kernel after: %s'
          % (strategy, len(bufs), len(bufs) * N / 1e9,
             'ok' if limit is None else limit, trivial_ok()))
    del bufs
    q.finish()
