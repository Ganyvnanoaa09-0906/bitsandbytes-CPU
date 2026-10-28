"""official_shape_check.py -- run CLBlast's own sample shape verbatim.

samples/sgemm.cpp uses m=128, n=64, k=512 with a_ld=k=512, b_ld=n=64, c_ld=n=64 --
a non-cubic shape, and one where a_ld (512) is much larger than m (128). CLBlast's
validation message is "LD of A is smaller than the matrix's first dimension", and
for A as (m,k) that first dimension is m, so the requirement is ld_a >= m -- which
is not the standard row-major convention (there ld_a would be k). The sample
satisfies it incidentally because k > m; my shapes had k < m, hence -1016
kInvalidLeadDimA every time.

If the sample's shape runs and computes correctly here, the rule is settled and the
model's leading dimensions can simply be set to satisfy it.

The sample uses alpha=0.7, beta=1.0, so C is initialised non-zero and must be
checked as alpha*A@B + beta*C_init.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import ctypes
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

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


def gemm(A, B, C0, m, n, k, lda, ldb, ldc, alpha=0.7, beta=1.0):
    ba = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A))
    bb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(B))
    bc = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(C0))
    rc = bl.CLBlastSgemm(ROW, NO, NO, m, n, k, ctypes.c_float(alpha),
                         ctypes.c_void_p(int(ba.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, ldb,
                         ctypes.c_float(beta),
                         ctypes.c_void_p(int(bc.int_ptr)), 0, ldc,
                         ctypes.byref(qp), None)
    queue.finish()
    out = None
    if rc == 0:
        out = np.zeros((m, n), dtype=np.float32)
        cl.enqueue_copy(queue, out, bc).wait()
    del ba, bb, bc
    return rc, out


print('=== 官方示例的原样参数: m=128 n=64 k=512, alpha=0.7 beta=1.0, ld=(512,64,64) ===')
rng = np.random.default_rng(2)
m, n, k = 128, 64, 512
A = rng.standard_normal((m, k)).astype(np.float32)
B = rng.standard_normal((k, n)).astype(np.float32)
C0 = rng.standard_normal((m, n)).astype(np.float32)
ref = 0.7 * (A @ B) + 1.0 * C0
rc, out = gemm(A, B, C0, m, n, k, k, n, n)
if rc != 0:
    print('  返回码 %d ⇒ 官方形状也失败 ✗' % rc)
else:
    d = float(np.abs(out - ref).max()) / float(np.abs(ref).max())
    print('  返回码 0，相对差 %.3g ⇒ %s' % (d, '✓ 正确' if d < 1e-4 else '✗ 结果不符'))
print()

print('=== 同样的数据、维度，只把 ld_a 从 k 改成 m（看校验规则 ✓）===')
for lda, ldb, ldc in ((k, n, n), (m, n, n), (m, k, m), (max(m, k), n, n)):
    rc, out = gemm(A, B, C0, m, n, k, lda, ldb, ldc)
    if rc != 0:
        print('  ld=(%d,%d,%d) ⇒ 返回码 %d ✗' % (lda, ldb, ldc, rc))
    else:
        d = float(np.abs(out - ref).max()) / float(np.abs(ref).max())
        print('  ld=(%d,%d,%d) ⇒ ✓ 启动，相对差 %.3g %s'
              % (lda, ldb, ldc, d, '✓ 正确' if d < 1e-4 else '✗ 不符'))
print()

print('=== 模型真实形状 M=1024 N=2736 K=512（用能通过校验的 ld ✓）===')
m2, n2, k2 = 256, 512, 128        # 缩小版，先确认规则
A2 = rng.standard_normal((m2, k2)).astype(np.float32)
B2 = rng.standard_normal((k2, n2)).astype(np.float32)
C2 = rng.standard_normal((m2, n2)).astype(np.float32)
ref2 = 0.7 * (A2 @ B2) + C2
for lda, ldb, ldc in ((max(m2, k2), n2, n2), (m2, max(n2, k2), m2), (max(m2, k2), max(n2, k2), max(m2, n2))):
    rc, out = gemm(A2, B2, C2, m2, n2, k2, lda, ldb, ldc)
    if rc != 0:
        print('  ld=(%d,%d,%d) ⇒ 返回码 %d ✗' % (lda, ldb, ldc, rc))
    else:
        d = float(np.abs(out - ref2).max()) / float(np.abs(ref2).max())
        print('  ld=(%d,%d,%d) ⇒ ✓ 启动，相对差 %.3g %s'
              % (lda, ldb, ldc, d, '✓ 正确' if d < 1e-4 else '✗ 不符'))
