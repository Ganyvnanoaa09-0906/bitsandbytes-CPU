"""MLP-only breakdown for the 33M AR step: GEMM vs master->device weight copies vs remainder.

Shapes are the actual 9 MLP GEMMs per layer (w1/w3/w2 fwd, dX, dW). SGEMM CLBlast, fp32.
The copy benchmark mirrors the training loop's per-layer master-slice -> shared weight buffer
copies. Remainder = MLP total (from layer split) - GEMM - copies.
"""
from __future__ import annotations

import ctypes
import statistics
import time

import numpy as np
import pyopencl as cl

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO, YES = 101, 111, 112
M, MD, FF = 2048, 512, 1368
L = 10

shapes = {
    'fwd_w1': (M, FF, MD), 'fwd_w3': (M, FF, MD), 'fwd_w2': (M, MD, FF),
    'bwd_w2_dx': (M, MD, FF), 'bwd_w1_dh': (M, FF, MD), 'bwd_w3_dh': (M, FF, MD),
    'bwd_w1_dw': (MD, FF, M), 'bwd_w3_dw': (MD, FF, M), 'bwd_w2_dw': (FF, MD, M),
}

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
bl = ctypes.CDLL(DLL)
bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))
print('device %s CU=%d' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))

bufs = {}
for name, (m, n, k) in shapes.items():
    a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros((m, k), np.float32))
    b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros((k, n), np.float32))
    c = cl.Buffer(ctx, mf.READ_WRITE, m * n * 4)
    bufs[name] = (a, b, c)


def call(name):
    m, n, k = shapes[name]
    a, b, c = bufs[name]
    rc = bl.CLBlastSgemm(ROW, NO, NO, m, n, k, 1.0,
                         ctypes.c_void_p(int(a.int_ptr)), 0, k,
                         ctypes.c_void_p(int(b.int_ptr)), 0, n, 0.0,
                         ctypes.c_void_p(int(c.int_ptr)), 0, n,
                         ctypes.byref(QP), None)
    if rc:
        raise RuntimeError(name, rc)


times = {}
for name in shapes:
    call(name); queue.finish()
    ts = []
    for _ in range(7):
        t0 = time.perf_counter(); call(name); queue.finish(); ts.append(time.perf_counter() - t0)
    times[name] = statistics.median(ts)

gemm_layer = sum(times.values())
gemm_step = gemm_layer * L
flops_step = sum(2 * m * n * k * L for (m, n, k) in shapes.values())
print('GEMM per layer %.2f ms  x10 = %.1f ms  (%.0f GFLOPS)' %
      (gemm_layer * 1e3, gemm_step * 1e3, flops_step / gemm_step / 1e9))
for name in shapes:
    m, n, k = shapes[name]
    print('  %-12s %6d %6d %6d  %.2f ms  %.0f GFLOPS' %
          (name, m, n, k, times[name] * 1e3, 2 * m * n * k / times[name] / 1e9))

# copy: master mlp weights -> shared device buffers, both fwd and bwd (2x per step)
n_mlp = MD * FF + MD * FF + FF * MD
master = cl.Buffer(ctx, mf.READ_WRITE, n_mlp * 4)
dst = cl.Buffer(ctx, mf.READ_WRITE, n_mlp * 4)
A = np.zeros(n_mlp, np.float32)
cl.enqueue_copy(queue, master, A)
queue.finish()
tc = []
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(L * 2):
        cl.enqueue_copy(queue, dst, master, byte_count=n_mlp * 4)
    queue.finish()
    tc.append(time.perf_counter() - t0)
copy_step = statistics.median(tc[1:])
print('copy MLP master->dev (10 layers x fwd+bwd): %.1f ms' % (copy_step * 1e3))
print('MLP total from formal split: fwd 576 + bwd 1058 = 1634 ms')
print('remainder (launch/sync/elementwise) = %.1f ms (%.0f%%)' %
      ((1.634 - gemm_step - copy_step) * 1e3,
       100 * (1.634 - gemm_step - copy_step) / 1.634))
