"""fp16 GEMM at Qwen-style layer shapes, to project what the iGPU could do on 1.7B / 8B LoRA steps.

The AR-shape measurement was 546-800 GFLOPS (mean ~731). LLM layers are larger (hidden 2048-12288),
so efficiency should be at least as good; this measures the actual layer shapes a LoRA step issues.
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ONE, ZERO = 0x3C00, 0x0000
LAYOUT_COL, TRANSPOSE_NO = 0, 111

SHAPES = {
    'Qwen3-1.7B qkv/o': (512, 2048, 2048),
    'Qwen3-1.7B mlp up': (512, 6144, 2048),
    'Qwen3-1.7B mlp down': (512, 2048, 6144),
    'Qwen3-8B qkv/o': (512, 4096, 4096),
    'Qwen3-8B mlp up': (512, 12288, 4096),
    'Qwen3-8B mlp down': (512, 4096, 12288),
}

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
bl = ctypes.CDLL(DLL)
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
qp = ctypes.c_void_p(int(queue.int_ptr))
print('device: %s (%d CU)' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))
print('%-24s %6s %6s %6s  %9s %9s  %7s' % ('shape', 'M', 'N', 'K', 'GFLOPS', 'ms', 'max|err|'))
for name, (m, n, k) in SHAPES.items():
    a = (np.random.randn(m, k) * 0.1).astype(np.float16)
    b = (np.random.randn(k, n) * 0.1).astype(np.float16)
    c = np.zeros((m, n), dtype=np.float16)
    abuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    bbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    cbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    def call():
        return bl.CLBlastHgemm(
            LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, ONE,
            ctypes.c_void_p(int(bbuf.int_ptr)), 0, n,
            ctypes.c_void_p(int(abuf.int_ptr)), 0, k, ZERO,
            ctypes.c_void_p(int(cbuf.int_ptr)), 0, n,
            ctypes.byref(qp), None)

    rc = call()
    queue.finish()
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        call()
        queue.finish()
        samples.append(time.perf_counter() - t0)
    sec = statistics.median(samples)
    gflops = 2.0 * m * n * k / sec / 1e9
    cl.enqueue_copy(queue, c, cbuf)
    queue.finish()
    ref = a.astype(np.float64) @ b.astype(np.float64)
    err = float(np.max(np.abs(c.astype(np.float64) - ref)))
    print('%-24s %6d %6d %6d  %9.1f %9.2f  %7.3f (rc=%d)'
          % (name, m, n, k, gflops, sec * 1e3, err, rc))
