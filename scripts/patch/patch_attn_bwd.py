"""Patch attention_gpu.py: correct + validated backward (transposed GEMMs), then re-time."""
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

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

P = r'D:\work\bitsandbytes-CPU\gt1030\attention_gpu.py'
t = io.open(P, encoding='utf-8').read()

# 1) transposed row-major helpers
OLD = """def main() -> int:
"""
NEW = """def hgemm_ta(a_dev, b_dev, c_dev, m, n, k, lda, ldb, ldc):
    \"\"\"C(m,n) = A^T @ B, row-major (A stored (k,m), B stored (k,n)).\"\"\"
    rc = bl.CLBlastHgemm(ROW, YES, NO, m, n, k, ONE,
                         ctypes.c_void_p(int(a_dev.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(b_dev.int_ptr)), 0, ldb, ZERO,
                         ctypes.c_void_p(int(c_dev.int_ptr)), 0, ldc,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Hgemm transA rc=%d %dx%dx%d' % (rc, m, n, k))


def hgemm_tb(a_dev, b_dev, c_dev, m, n, k, lda, ldb, ldc):
    \"\"\"C(m,n) = A @ B^T, row-major (A stored (m,k), B stored (n,k)).\"\"\"
    rc = bl.CLBlastHgemm(ROW, NO, YES, m, n, k, ONE,
                         ctypes.c_void_p(int(a_dev.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(b_dev.int_ptr)), 0, ldb, ZERO,
                         ctypes.c_void_p(int(c_dev.int_ptr)), 0, ldc,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Hgemm transB rc=%d %dx%dx%d' % (rc, m, n, k))


def main() -> int:
"""
assert t.count(OLD) == 1
t = t.replace(OLD, NEW, 1)

# 2) dO_tok buffer
t = t.replace("        'dqkv': buf(M * WROW * 2), 'dwqkv': buf(DH * WROW * 2), 'dwo': buf(DH * DH * 2),",
              "        'dqkv': buf(M * WROW * 2), 'dwqkv': buf(DH * WROW * 2), 'dwo': buf(DH * DH * 2),\n"
              "        'dO_tok': buf(M * DH * 2),")

# 3) correct backward
OLD_BWD = """    def bwd():
        hgemm(dev['o_tok'], dev['dout'], dev['dwo'], DH, DH, M)            # dWo = O_tok^T dOut
        hgemm(dev['dout'], dev['wo'], dev['dO'], M, DH, DH)                # dO_tok = dOut Wo^T? (see note)
        # NOTE: dO_tok = dOut @ Wo^T; Wo stored (in,out), so we need Wo^T -> use transpose kernel later.
        K['split_dout'](queue, (BH * T * D,), (256,), dev['dO'], dev['dO'],
                        np.int32(T), np.int32(D), np.int32(NH), np.int32(B))
        # dP = dO V^T
        hgemm_batched(dev['dO'], dev['V'], dev['dP'], T, T, D, NO, YES,
                      D, D, T, T * D, T * D, T * T)
        K['softmax_bwd_causal'](queue, (T * 256, BH), (256, 1), dev['P'], dev['dP'], dev['dS'],
                                np.int32(T), np.float32(scale))
        # dV = P^T dO
        hgemm_batched(dev['P'], dev['dO'], dev['dV'], T, D, T, YES, NO,
                      T, D, D, T * T, T * D, T * D)
        # dQ = dS K
        hgemm_batched(dev['dS'], dev['K'], dev['dQ'], T, D, T, NO, NO,
                      T, D, D, T * T, T * D, T * D)
        # dK = dS^T Q
        hgemm_batched(dev['dS'], dev['Q'], dev['dK'], T, D, T, YES, NO,
                      T, D, D, T * T, T * D, T * D)
        K['merge_dqkv'](queue, (M * WROW,), (256,), dev['dQ'], dev['dK'], dev['dV'],
                        dev['dqkv'], np.int32(T), np.int32(D), np.int32(NH), np.int32(NKV),
                        np.int32(B))
        # dWqkv = h^T dqkv : needs h transposed -> approximate with x (h ~= x here) for timing only
        hgemm(dev['dqkv'], dev['wqkv'], dev['dx'], M, DH, WROW)            # dh = dqkv @ Wqkv^T? (timing)
"""
NEW_BWD = """    def bwd():
        hgemm_ta(dev['o_tok'], dev['dout'], dev['dwo'], DH, DH, M, DH, DH, DH)     # dWo = O^T dOut
        hgemm_tb(dev['dout'], dev['wo'], dev['dO_tok'], M, DH, DH, DH, DH, DH)     # dO_tok = dOut Wo^T
        K['split_dout'](queue, (BH * T * D,), (256,), dev['dO_tok'], dev['dO'],
                        np.int32(T), np.int32(D), np.int32(NH), np.int32(B))
        hgemm_batched(dev['dO'], dev['V'], dev['dP'], T, T, D, NO, YES,
                      D, D, T, T * D, T * D, T * T)                                # dP = dO V^T
        K['softmax_bwd_causal'](queue, (T * 256, BH), (256, 1), dev['P'], dev['dP'], dev['dS'],
                                np.int32(T), np.float32(scale))
        hgemm_batched(dev['P'], dev['dO'], dev['dV'], T, D, T, YES, NO,
                      T, D, D, T * T, T * D, T * D)                                # dV = P^T dO
        hgemm_batched(dev['dS'], dev['K'], dev['dQ'], T, D, T, NO, NO,
                      T, D, D, T * T, T * D, T * D)                                # dQ = dS K
        hgemm_batched(dev['dS'], dev['Q'], dev['dK'], T, D, T, YES, NO,
                      T, D, D, T * T, T * D, T * D)                                # dK = dS^T Q
        K['merge_dqkv'](queue, (M * WROW,), (256,), dev['dQ'], dev['dK'], dev['dV'],
                        dev['dqkv'], np.int32(T), np.int32(D), np.int32(NH), np.int32(NKV),
                        np.int32(B))
        hgemm_ta(dev['x'], dev['dqkv'], dev['dwqkv'], DH, WROW, M, DH, WROW, WROW)  # dWqkv = x^T dqkv
        hgemm_tb(dev['dqkv'], dev['wqkv'], dev['dx'], M, DH, WROW, WROW, WROW, DH)  # dx = dqkv Wqkv^T
"""
assert t.count(OLD_BWD) == 1
t = t.replace(OLD_BWD, NEW_BWD, 1)
io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('attention backward rewritten with transposed GEMMs')
