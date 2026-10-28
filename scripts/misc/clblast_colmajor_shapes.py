"""Can CLBlast do non-cubic GEMM on this iGPU? The report left this open (CLBlast -1011/-1016 on
every non-square shape, independent of size, so it is a call-convention problem and not a size one).

Fix under test: the row-major API path is the one that fails. Call the column-major API instead and
compute the transposed product, which is a pure index mapping and needs no copies:

    want  C_row (M x N) = A_row (M x K) @ B_row (K x N)
    in column-major terms:  C' (N x M) = B_row^T (N x K) @ A_row^T (K x M)
    so call Sgemm(layout=COL, transA=NO, transB=NO, m=N, n=M, k=K,
                  A = B_row memory, a_ld = N,   B = A_row memory, b_ld = K,
                  C = C_row memory, c_ld = N)

Correctness is checked against numpy on non-cubic shapes before any timing, then throughput on the
shapes the AR model actually uses.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import ctypes
import os
import statistics as st
import sys
import time

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
if not os.path.exists(DLL):
    raise SystemExit('missing %s' % DLL)

bl = ctypes.CDLL(DLL)
bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
]
LAYOUT_COL, TRANSPOSE_NO = 0, 111

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(queue.int_ptr))
print('device: %s | %d CU | peak fp32 1150 GFLOPS' % (dev.name, dev.max_compute_units))


def run(m, n, k, a, b, c, queue_ptr=None):
    """C_row[m,n] = A_row[m,k] @ B_row[k,n], via the column-major API on the same buffers."""
    return bl.CLBlastSgemm(
        LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO,
        n, m, k, ctypes.c_float(1.0),
        ctypes.c_void_p(int(b.int_ptr)), 0, n,
        ctypes.c_void_p(int(a.int_ptr)), 0, k,
        ctypes.c_float(0.0),
        ctypes.c_void_p(int(c.int_ptr)), 0, n,
        ctypes.byref(queue_ptr or qp), None)


def bench(m, n, k, iters=5, warm=2, label='', check=False):
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    C = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * 4)

    def once():
        rc = run(m, n, k, A, B, C)
        if rc != 0:
            raise RuntimeError('CLBlast rc=%d on %s' % (rc, label or (m, n, k)))
        queue.finish()

    for _ in range(warm):
        once()
    if check:
        out = np.empty((m, n), np.float32)
        cl.enqueue_copy(queue, out, C).wait()
        ref = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
        err = float(np.max(np.abs(out - ref)))
        rel = err / (float(np.max(np.abs(ref))) + 1e-30)
        print('  correctness %-16s max_abs_err=%.3e rel=%.3e  %s'
              % (label, err, rel, 'OK' if rel < 1e-4 else 'WRONG'))
        if rel >= 1e-4:
            raise SystemExit('col-major trick produces the wrong product; stop here')

    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        once()
        ts.append(time.perf_counter() - t0)
    dt = st.median(ts)
    gf = 2.0 * m * n * k / dt / 1e9
    print('  igpu  %-18s m=%-5d n=%-5d k=%-5d  %8.3f ms  %7.1f GFLOPS  %5.1f%%'
          % (label, m, n, k, dt * 1e3, gf, 100 * gf / 1150.0))
    return gf, dt


print()
print('=== correctness first (non-cubic) ===')
for (m, n, k) in ((3, 5, 7), (17, 23, 31), (128, 96, 64)):
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    C = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * 4)
    rc = run(m, n, k, A, B, C)
    if rc != 0:
        print('  rc=%d on (%d,%d,%d)' % (rc, m, n, k))
        raise SystemExit('col-major path also fails; conclusion changes')
    queue.finish()
    out = np.empty((m, n), np.float32)
    cl.enqueue_copy(queue, out, C).wait()
    ref = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    err = float(np.max(np.abs(out - ref)))
    print('  (%3d,%3d,%3d)  max_abs_err=%.3e  %s' % (m, n, k, err, 'OK' if err < 1e-3 else 'WRONG'))
    if err >= 1e-3:
        raise SystemExit('wrong product')

print()
print('=== warm-up (iGPU clocks need >15 s of sustained load) ===')
a = np.random.randn(1024, 1024).astype(np.float32)
b = np.random.randn(1024, 1024).astype(np.float32)
A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
C = cl.Buffer(ctx, mf.WRITE_ONLY, 1024 * 1024 * 4)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 15:
    run(1024, 1024, 1024, A, B, C)
    queue.finish()
print('  warm-up done')

SHAPES = [
    ('MLP up    ', 2048, 2048, 512),
    ('MLP down  ', 2048, 512, 2048),
    ('qkv proj  ', 2048, 1536, 512),
    ('o proj    ', 2048, 512, 512),
    ('head      ', 2048, 1024, 512),
    ('attn QK   ', 2048, 1024, 64),
    ('attn AV   ', 2048, 64, 1024),
    ('cube 1024 ', 1024, 1024, 1024),
    ('cube 2048 ', 2048, 2048, 2048),
]
print()
print('=== CLBlast (column-major API), real AR shapes ===')
rows = []
for label, m, n, k in SHAPES:
    gf, dt = bench(m, n, k, iters=5, warm=2, label=label,
                   check=(label.strip().startswith('MLP up') or label.strip().startswith('o proj')))
    rows.append((label.strip(), m, n, k, gf, dt))

# CPU comparison, same shapes, same session
print()
print('=== CPU torch fp32 (6 threads), same shapes ===')
import torch
torch.set_num_threads(6)
torch.set_grad_enabled(False)
print('  torch %s' % torch.__version__)
for label, m, n, k in SHAPES:
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    for _ in range(2):
        a @ b
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        a @ b
        ts.append(time.perf_counter() - t0)
    dt = st.median(ts)
    gf = 2.0 * m * n * k / dt / 1e9
    print('  cpu   %-18s m=%-5d n=%-5d k=%-5d  %8.3f ms  %7.1f GFLOPS  %5.1f%%'
          % (label, m, n, k, dt * 1e3, gf, 100 * gf / 456.0))

print()
print('=== ratio (CPU / iGPU) on the same shape, from the two runs above ===')
