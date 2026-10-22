"""AR-30.5M step GEMM phase with CLBlast SGEMM (fp32), same shapes/schedule as the Hgemm run.

Answers the precision/performance trade: if Sgemm effective throughput is >=300 GFLOPS, the
GEMM phase is ~1.2 s for 364 GFLOP, so an fp32-accumulation prototype is ~2x CPU (3.56 s).
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np
import pyopencl as cl

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
LAYOUT_COL, TRANSPOSE_NO = 0, 111
M, D, FF, VOCAB = 2048, 512, 2048, 1024
TSEQ, L = 256, 10

schedule = []
for _ in range(L):
    schedule += [('fwd qkv', M, 1024, D, 1), ('fwd attn QK', M, TSEQ, 64, 1),
                 ('fwd attn AV', M, 64, TSEQ, 1), ('fwd out', M, D, D, 1),
                 ('fwd mlp-up', M, FF, D, 1), ('fwd mlp-down', M, D, FF, 1)]
    schedule += [('bwd qkv dX', M, D, 1024, 1), ('bwd qkv dW', D, 1024, M, 1),
                 ('bwd out dX', M, D, D, 1), ('bwd out dW', D, D, M, 1),
                 ('bwd mlp-up dX', M, D, FF, 1), ('bwd mlp-up dW', D, FF, M, 1),
                 ('bwd mlp-down dX', M, FF, D, 1), ('bwd mlp-down dW', FF, D, M, 1),
                 ('bwd attn dQ', M, 64, TSEQ, 1), ('bwd attn dK', M, 64, TSEQ, 1),
                 ('bwd attn dV', M, TSEQ, 64, 1)]
schedule += [('fwd head', M, VOCAB, D, 1), ('bwd head dX', M, D, VOCAB, 1),
             ('bwd head dW', D, VOCAB, M, 1)]

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
qp = ctypes.c_void_p(int(queue.int_ptr))
print('device: %s (%d CU)' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))

unique = sorted({(m, n, k) for _l, m, n, k, _c in schedule})
bufs = {}
for (m, n, k) in unique:
    a = np.zeros((m, k), dtype=np.float32)
    b = np.zeros((k, n), dtype=np.float32)
    c = np.zeros((m, n), dtype=np.float32)
    bufs[(m, n, k)] = (
        cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
        cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b),
        cl.Buffer(ctx, mf.READ_ONLY, c.nbytes),
    )


def call(m, n, k):
    ab, bb, cb = bufs[(m, n, k)]
    return bl.CLBlastSgemm(
        LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, 1.0,
        ctypes.c_void_p(int(bb.int_ptr)), 0, n,
        ctypes.c_void_p(int(ab.int_ptr)), 0, k, 0.0,
        ctypes.c_void_p(int(cb.int_ptr)), 0, n,
        ctypes.byref(qp), None)


per_shape = {}
for (m, n, k) in unique:
    call(m, n, k); queue.finish()
    samples = []
    for _ in range(7):
        t0 = time.perf_counter(); call(m, n, k); queue.finish(); samples.append(time.perf_counter() - t0)
    per_shape[(m, n, k)] = statistics.median(samples)

total = 0.0
print()
print('%-16s %6s %6s %6s %5s %10s %9s' % ('gemm', 'M', 'N', 'K', 'cnt', 'per-call', 'GFLOPS'))
for label, m, n, k, cnt in schedule:
    sec = per_shape[(m, n, k)]
    total += sec * cnt
    if cnt == 1 and label in ('fwd qkv', 'fwd mlp-up', 'bwd qkv dW', 'bwd mlp-up dW'):
        print('%-16s %6d %6d %6d %5d %8.2f ms %9.1f'
              % (label, m, n, k, cnt, sec * 1e3, 2.0 * m * n * k / sec / 1e9))
step_flops = sum(2.0 * m * n * k * c for _l, m, n, k, c in schedule)
print()
print('SGEMM GEMM calls/step: %d' % sum(c for *_x, c in schedule))
print('SGEMM FLOPs/step    : %.1f GFLOP' % (step_flops / 1e9))
print('SGEMM GEMM phase    : %.3f s  (effective %.0f GFLOPS)' % (total, step_flops / total / 1e9))
fwd_labels = [x for x in schedule if x[0].startswith('fwd')]
bwd_labels = [x for x in schedule if x[0].startswith('bwd')]
attn_labels = [x for x in schedule if 'attn' in x[0]]
def subtotal(labels):
    return sum(per_shape[(m,n,k)]*c for _l,m,n,k,c in labels)
print('SGEMM fwd  : %.3f s' % subtotal(fwd_labels))
print('SGEMM bwd  : %.3f s' % subtotal(bwd_labels))
print('SGEMM attn : %.3f s' % subtotal(attn_labels))
print('fraction of 0.594 s : %.2fx' % (0.594 / total))
print('whole-step rough    : %.2f s (GEMM + ~0.1-0.3s CPU phase + overhead)' % (total + 0.2))
print('vs CPU 3.56s        : %.2fx' % (3.56 / (total + 0.2)))
