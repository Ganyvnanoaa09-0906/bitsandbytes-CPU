"""what_does_clblast_compute.py -- identify the actual output layout, then fix the call.

The evidence so far is contradictory in a way that has a narrow set of explanations:

  * diag_clblast_layout.py: 8x8x8, max diff 0, closest match "B@A"
  * tiny_clblast.py:        8x8x8, max diff 28   (but it ran a failing 4x3x2 first)
  * verify_clblast_correct.py: 256..2048 cubic, relative error 1.30-1.60

"Closest match B@A" is the useful clue. Swapping or transposing operands does not
change the amount of arithmetic, only how memory is read -- so if CLBlast is
computing a permuted product, the throughput numbers may still be valid and the
call convention is simply wrong. For random square matrices A@B and (A@B).T differ
by O(1) relative, which is what the 1.30-1.60 figures look like.

So: use asymmetric operands, no failing calls before them, and compare the output
against every plausible permutation. Then search the (layout, transA, transB,
leading dimension) space for the combination that reproduces A@B exactly.
"""
from __future__ import annotations

import ctypes
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'

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
import pyopencl as cl  # noqa: E402

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(queue.int_ptr))

ROW, COL, NO, YES = 1, 2, 111, 112

m, n, k = 6, 4, 5          # deliberately asymmetric: m != n != k
rng = np.random.default_rng(7)
A = rng.standard_normal((m, k)).astype(np.float32)
B = rng.standard_normal((k, n)).astype(np.float32)
ref = A @ B
print('A %s  B %s  ⇒  A@B %s   （非对称，避免巧合相等）' % (A.shape, B.shape, ref.shape))
print()


def call(layout, ta, tb, lda, ldb, ldc, out_shape):
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.READ_WRITE, int(np.prod(out_shape)) * 4)
    zeros = np.zeros(out_shape, dtype=np.float32)
    cl.enqueue_copy(queue, bc, zeros).wait()
    rc = bl.CLBlastSgemm(layout, ta, tb, m, n, k, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(ba.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, ldb,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(bc.int_ptr)), 0, ldc,
                         ctypes.byref(qp), None)
    queue.finish()
    out = np.zeros(out_shape, dtype=np.float32)
    if rc == 0:
        cl.enqueue_copy(queue, out, bc).wait()
    return rc, out


# 1) what does the current call actually produce?
rc, got = call(ROW, NO, NO, k, n, n, (m, n))
print('当前调用 (row, NO, NO, ld=k,n,n) 返回码 %d' % rc)
if rc == 0:
    cands = {
        'A@B': ref,
        '(A@B).T': ref.T,
        'B.T@A.T': B.T @ A.T,
        'A.T@B': A.T @ B if A.T.shape[1] == B.shape[0] else None,
        'A@B.T': A @ B.T if A.shape[1] == B.T.shape[0] else None,
    }
    best, bd = None, 1e9
    for nm, c in cands.items():
        if c is None or c.shape != got.shape:
            continue
        d = float(np.abs(got - c).max())
        print('   与 %-10s 最大差 %10.4g' % (nm, d))
        if d < bd:
            best, bd = nm, d
    print('   ⇒ 实际产出 = 【%s】（差 %.3g）' % (best, bd))
print()

# 2) search for a combination that gives A@B in row-major C
print('搜索能算出 A@B 的参数组合:')
found = []
for layout, lname in ((ROW, 'row'), (COL, 'col')):
    for ta, tan in ((NO, 'N'), (YES, 'T')):
        for tb, tbn in ((NO, 'N'), (YES, 'T')):
            for lda in (k, m):
                for ldb in (n, k):
                    for ldc in (n, m):
                        try:
                            rc, out = call(layout, ta, tb, lda, ldb, ldc, (m, n))
                        except Exception:
                            continue
                        if rc != 0:
                            continue
                        if np.allclose(out, ref, atol=1e-3, rtol=1e-4):
                            tag = '%s A=%s B=%s ld=%d,%d,%d' % (lname, tan, tbn, lda, ldb, ldc)
                            found.append(tag)
                            print('   ✓ 命中: %s' % tag)
print()
if found:
    print('  ⇒ 用【%s】即可正确计算 A@B ✓' % found[0])
else:
    print('  ⇒ 没有组合能算出 A@B ✗ ⇒ 不是调用约定问题，而是 CLBlast 在本设备上算错')
