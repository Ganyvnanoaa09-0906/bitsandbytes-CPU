"""P1a first number: the AR-30.5M training step's GEMM content on the iGPU, phase-separated.

This is the compute schedule of one real training step (forward + backward, full model), with each
GEMM issued at the shape the model actually uses, measured on the iGPU with fp16 CLBlast. It is NOT
yet a full training step (no activations/softmax/optimizer, no CPU-GPU transfers); it answers the one
question that decides the next week of work: after "phase separation", how much of the 0.594 s GT 1030
reference does the iGPU GEMM phase alone account for?

Model: d_model 512, 10 layers, n_head 8, n_kv_head 4 (head_dim 64), ffn 2048, vocab 1024, tokens/step
= 2048 (T=256 x batch 8, the config behind the 375 GFLOP/step figure in report 7.12).

For each linear weight W (d_in -> d_out) the step needs three GEMMs:
  forward  (M, d_out, d_in)        Y  = X  @ W
  dX       (M, d_in,  d_out)       dX = dY @ W^T
  dW       (d_in, d_out, M)        dW = X^T @ dY     (accumulated for the optimizer)
Attention adds QK^T, P@V forward and their three backward GEMMs per layer.
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

M = 2048          # tokens per step (T=256 x batch 8)
D = 512
FF = 2048
VOCAB = 1024
KV = 256          # n_kv_head 4 * head_dim 64
L = 10
TSEQ = 256        # per-head sequence length (tokens / batch = 256)

# ---- GEMM schedule for one step ----
# (label, m, n, k, count)
schedule = []
for _ in range(L):
    # forward
    schedule += [('fwd qkv', M, 1024, D, 1), ('fwd attn QK', M, TSEQ, 64, 1),
                 ('fwd attn AV', M, 64, TSEQ, 1), ('fwd out', M, D, D, 1),
                 ('fwd mlp-up', M, FF, D, 1), ('fwd mlp-down', M, D, FF, 1)]
    # backward (dX and dW for every linear)
    schedule += [('bwd qkv dX', M, D, 1024, 1), ('bwd qkv dW', D, 1024, M, 1),
                 ('bwd out dX', M, D, D, 1), ('bwd out dW', D, D, M, 1),
                 ('bwd mlp-up dX', M, D, FF, 1), ('bwd mlp-up dW', D, FF, M, 1),
                 ('bwd mlp-down dX', M, FF, D, 1), ('bwd mlp-down dW', FF, D, M, 1),
                 ('bwd attn dQ', M, 64, TSEQ, 1), ('bwd attn dK', M, 64, TSEQ, 1),
                 ('bwd attn dV', M, TSEQ, 64, 1)]
# head
schedule += [('fwd head', M, VOCAB, D, 1), ('bwd head dX', M, D, VOCAB, 1),
             ('bwd head dW', D, VOCAB, M, 1)]

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

# one set of buffers per unique shape (reused; only the GEMM shape matters for timing)
unique = sorted({(m, n, k) for _l, m, n, k, _c in schedule})
bufs = {}
for (m, n, k) in unique:
    a = np.zeros((m, k), dtype=np.float16)
    b = np.zeros((k, n), dtype=np.float16)
    c = np.zeros((m, n), dtype=np.float16)
    bufs[(m, n, k)] = (
        cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
        cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b),
        cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c),
    )


def call(m, n, k):
    ab, bb, cb = bufs[(m, n, k)]
    return bl.CLBlastHgemm(
        LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, ONE,
        ctypes.c_void_p(int(bb.int_ptr)), 0, n,
        ctypes.c_void_p(int(ab.int_ptr)), 0, k, ZERO,
        ctypes.c_void_p(int(cb.int_ptr)), 0, n,
        ctypes.byref(qp), None)


# measure each unique shape
per_shape = {}
for (m, n, k) in unique:
    call(m, n, k)          # warm
    queue.finish()
    samples = []
    for _ in range(7):
        t0 = time.perf_counter()
        call(m, n, k)
        queue.finish()
        samples.append(time.perf_counter() - t0)
    per_shape[(m, n, k)] = statistics.median(samples)

total = 0.0
gflops_total = 0.0
print()
print('%-16s %6s %6s %6s %5s %9s %9s' % ('gemm', 'M', 'N', 'K', 'cnt', 'per-call', 'GFLOPS'))
for label, m, n, k, cnt in schedule:
    sec = per_shape[(m, n, k)]
    total += sec * cnt
    gflops = 2.0 * m * n * k / sec / 1e9
    gflops_total += gflops * sec * cnt
    if cnt == 1 and label in ('fwd qkv', 'fwd mlp-up', 'bwd qkv dW', 'bwd mlp-up dW'):
        print('%-16s %6d %6d %6d %5d %8.2f ms %9.1f' % (label, m, n, k, cnt, sec * 1e3, gflops))

step_flops = sum(2.0 * m * n * k * c for _l, m, n, k, c in schedule)
print()
print('GEMM calls per step : %d' % sum(c for *_x, c in schedule))
print('GEMM FLOPs per step : %.1f GFLOP' % (step_flops / 1e9))
print('iGPU GEMM phase     : %.3f s   (effective %.0f GFLOPS)' % (total, step_flops / total / 1e9))
print('GT 1030 reference   : 0.594 s')
print('fraction (GEMM only): %.2fx' % (0.594 / total))
print()
print('reminder: this is the GEMM phase only. CPU-side softmax/norm/GELU/optimizer and the')
print('phase switches add time; the CPU phase was measured at ~3-9% of the 2.854 s CPU step')
print('(report 7.12), i.e. 0.09-0.26 s, so a whole-step projection is GEMM + CPU phase + overhead.')
