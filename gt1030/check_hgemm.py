"""Small-case correctness check for the fp16 Hgemm call used in bench_hgemm.py.

The AR-shape benchmark produced plausible GFLOPS but |err| of 8-17 against a float64 reference of
the same fp16 inputs, which cannot come from fp16 accumulation over K<=2048. This isolates the call
convention with tiny matrices where row-major and column-major interpretations are far apart.
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
bl = ctypes.CDLL(DLL)
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(queue.int_ptr))
ONE = 0x3C00
ZERO = 0

# simple integer-valued matrices so fp16 is exact
m, n, k = 2, 3, 2
A = np.array([[1, 2], [3, 4]], dtype=np.float16)          # MxK row-major
B = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float16)   # KxN row-major
C_expected = (A.astype(np.float64) @ B.astype(np.float64))
print('A=\n', A, '\nB=\n', B, '\nexpected=\n', C_expected)


def run(beta_one: bool, layout_choices):
    abuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    cbuf = cl.Buffer(ctx, mf.READ_WRITE, m * n * 2)
    zero_c = np.zeros((m, n), dtype=np.float16)
    cl.enqueue_copy(queue, cbuf, zero_c)
    queue.finish()
    transA, transB, mm, nn, kk = layout_choices
    beta = ONE if beta_one else 0
    # row-major API: C = A @ B directly
    rc = bl.CLBlastHgemm(0, transA, transB, mm, nn, kk, ONE,
                         ctypes.c_void_p(int(abuf.int_ptr)), 0, kk,
                         ctypes.c_void_p(int(bbuf.int_ptr)), 0, nn,
                         beta,
                         ctypes.c_void_p(int(cbuf.int_ptr)), 0, nn,
                         ctypes.byref(qp), None)
    queue.finish()
    out = np.empty((m, n), dtype=np.float16)
    cl.enqueue_copy(queue, out, cbuf)
    queue.finish()
    return rc, out


for name, choices in [('rowmajor m,n,k', (111, 111, m, n, k)),
                      ('colmajor trick', (111, 111, n, m, k))]:
    rc, out = run(True, choices)
    print('%-16s rc=%d out=\n%s' % (name, rc, out))
    if name == 'colmajor trick':
        print('  (colmajor path returns C^T if correct: expected T =', C_expected.T, ')')
