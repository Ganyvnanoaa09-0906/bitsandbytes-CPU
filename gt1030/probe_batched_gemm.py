"""Probe CLBlastHgemmStridedBatched: signature, layout convention, correctness.

If this works, the whole attention stack can run as 6 batched GEMM launches (QK, PV, dP, dV, dQ, dK)
over 64 (batch x head) matrices instead of thousands of per-head calls.
"""
from __future__ import annotations

import ctypes
import sys

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ONE, ZERO = 0x3C00, 0x0000
ROW, COL = 101, 102      # CLBlastLayout enum values from clblast_c.h
NO, TRANS = 111, 112

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
lib = ctypes.CDLL(DLL)

# CLBlast C API for HgemmStridedBatched: scalars are half (uint16), sizes size_t, queued via cl 1.2 pointers
lib.CLBlastHgemmStridedBatched.restype = ctypes.c_int
lib.CLBlastHgemmStridedBatched.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))

B, T, D = 8, 16, 8   # batch entries, matrix size, head dim
rng = np.random.default_rng(0)
Qh = (rng.standard_normal((B, T, D)).astype(np.float32) * 0.2).astype(np.float16)  # (B,T,D) per batch
ref = np.einsum('btd,bsd->bts', Qh.astype(np.float32), Qh.astype(np.float32))     # S = Q Q^T
print('reference S shape', ref.shape)

qb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Qh)
sb = cl.Buffer(ctx, mf.WRITE_ONLY, B * T * T * 2)


def run(layout, trans_a, trans_b, lda, ldb, ldc):
    rc = lib.CLBlastHgemmStridedBatched(
        layout, trans_a, trans_b, T, T, D,
        ONE,
        ctypes.c_void_p(int(qb.int_ptr)), 0, lda, T * D,
        ctypes.c_void_p(int(qb.int_ptr)), 0, ldb, T * D,
        ZERO,
        ctypes.c_void_p(int(sb.int_ptr)), 0, ldc, T * T,
        B, ctypes.byref(QP), None)
    queue.finish()
    return rc


for name, layout, ta, tb, lda, ldb, ldc in (
        ('ROW no/trans', ROW, NO, TRANS, D, D, T),
        ('ROW trans/no', ROW, TRANS, NO, T, T, T),
        ('COL no/trans', COL, NO, TRANS, T, D, T),
        ('COL trans/no', COL, TRANS, NO, T, T, T)):
    rc = run(layout, ta, tb, lda, ldb, ldc)
    got = np.empty((B, T, T), dtype=np.float16)
    cl.enqueue_copy(queue, got, sb)
    queue.finish()
    err = np.abs(got.astype(np.float32) - ref).max()
    print('%-12s rc=%d  max|err|=%.4f' % (name, rc, err))
