"""identity_clblast_style.py -- A = I makes the expected output unambiguous.

Two independent implementations -- CLBlast and the project's own OpenCL kernels --
both report a relative error of 1.3-1.6 against numpy. Two unrelated GEMMs do not
fail identically, so the shared factor is the verification, not the kernels. My
harness, and the check I just added to bench_opencl3.py, share the same comparison
code, which is exactly the kind of shared assumption that produces a shared wrong
answer.

A = identity removes every ambiguity: C must equal B exactly, element for element,
whatever the layout convention. If the kernel is right, C == B and my earlier
comparisons were wrong. If C != B, the kernels really are broken.
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
ROW, NO = 1, 111

n = 64
B = np.arange(n * n, dtype=np.float32).reshape(n, n) % 97
I = np.eye(n, dtype=np.float32)          # A = I  =>  C 必须等于 B

ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=I)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
bc = cl.Buffer(ctx, mf.READ_WRITE, n * n * 4)
zeros = np.zeros((n, n), dtype=np.float32)
cl.enqueue_copy(queue, bc, zeros).wait()

rc = bl.CLBlastSgemm(ROW, NO, NO, n, n, n, ctypes.c_float(1.0),
                     ctypes.c_void_p(int(ba.int_ptr)), 0, n,
                     ctypes.c_void_p(int(bb.int_ptr)), 0, n,
                     ctypes.c_float(0.0),
                     ctypes.c_void_p(int(bc.int_ptr)), 0, n,
                     ctypes.byref(qp), None)
queue.finish()
print('CLBlast I @ B,  I=单位阵, B[i,j]=(i*64+j)%%97,  n=%d, 返回码 %d' % (n, rc))

# read back two ways: as a (n,n) array and as a flat buffer, to catch a shape bug
out2d = np.zeros((n, n), dtype=np.float32)
cl.enqueue_copy(queue, out2d, bc).wait()
flat = np.zeros(n * n, dtype=np.float32)
cl.enqueue_copy(queue, flat, bc).wait()

print('  回读为 (n,n): 与 B 最大差 %.4g   与 B.T 最大差 %.4g'
      % (np.abs(out2d - B).max(), np.abs(out2d - B.T).max()))
print('  回读为 flat : 与 B.ravel() 最大差 %.4g' % np.abs(flat - B.ravel()).max())
print()
print('  C[0,:6] =', np.round(out2d[0, :6], 1))
print('  B[0,:6] =', np.round(B[0, :6], 1))
print('  C[:6,0] =', np.round(out2d[:6, 0], 1))
print('  B[:6,0] =', np.round(B[:6, 0], 1))
print()
if np.abs(out2d - B).max() < 1e-3:
    print('  ⇒ 【C == B ✓ 内核是对的 ✗ 我之前的对比代码有错 ✓✓】')
elif np.abs(out2d - B.T).max() < 1e-3:
    print('  ⇒ 【C == B.T ✓ 内核算的是转置 ⇒ 布局约定问题 ✓】')
else:
    print('  ⇒ 【C 既不是 B 也不是 B.T ⇒ 内核确实算错 ✗✗】')
