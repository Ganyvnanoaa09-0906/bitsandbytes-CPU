"""Minimal iGPU helper for 8B hybrid training: transpose kernel + CLBlast Sgemm/Hgemm.

Used by the autograd Function in train_qwen3_8b_nf4_lora_hybrid.py.
Buffers are cached by byte size to avoid repeated cl_mem creation (gfx902/WDDM).
"""
from __future__ import annotations

import ctypes

import numpy as np
import pyopencl as cl

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO = 101, 111

TRANSPOSE = r'''
#define TILE 32
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void transpose_f32(__global const float* restrict src, __global float* restrict dst,
                   const int rows, const int cols) {
    __local float tile[TILE][TILE + 1];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int x0 = get_group_id(0) * TILE, y0 = get_group_id(1) * TILE;
    if (x0 + tx < cols && y0 + ty < rows)
        tile[ty][tx] = src[(y0 + ty) * cols + (x0 + tx)];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x0 + tx < cols && y0 + ty < rows)
        dst[(x0 + tx) * rows + (y0 + ty)] = tile[ty][tx];
}
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void transpose_f16(__global const half* restrict src, __global half* restrict dst,
                   const int rows, const int cols) {
    __local half tile[TILE][TILE + 1];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int x0 = get_group_id(0) * TILE, y0 = get_group_id(1) * TILE;
    if (x0 + tx < cols && y0 + ty < rows)
        tile[ty][tx] = src[(y0 + ty) * cols + (x0 + tx)];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x0 + tx < cols && y0 + ty < rows)
        dst[(x0 + tx) * rows + (y0 + ty)] = tile[ty][tx];
}
'''


_ctx = None
_q = None
_mf = None
_kern = None
_bl = None
_cache = {}


def init():
    global _ctx, _q, _mf, _kern, _bl
    if _ctx is not None:
        return
    _ctx = cl.create_some_context(interactive=False)
    _q = cl.CommandQueue(_ctx)
    _mf = cl.mem_flags
    prog = cl.Program(_ctx, TRANSPOSE).build(options=['-cl-mad-enable'])
    _kern = {n: cl.Kernel(prog, n) for n in ('transpose_f32', 'transpose_f16')}
    _bl = ctypes.CDLL(DLL)
    _bl.CLBlastSgemm.restype = ctypes.c_int
    _bl.CLBlastSgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    _bl.CLBlastHgemm.restype = ctypes.c_int
    _bl.CLBlastHgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ushort,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ushort,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]


def _buf(nbytes, slot=0):
    init()
    key = (nbytes, slot)
    if key not in _cache:
        _cache[key] = cl.Buffer(_ctx, _mf.READ_WRITE, nbytes)
    return _cache[key]


def _ptr(b):
    return ctypes.c_void_p(int(b.int_ptr))


def transpose_f16(src: np.ndarray) -> np.ndarray:
    """src (rows, cols) fp16 -> (cols, rows) fp16 on iGPU."""
    init()
    rows, cols = src.shape
    n = rows * cols * 2
    sb = _buf(n, 0)
    db = _buf(n, 1)
    cl.enqueue_copy(_q, sb, np.ascontiguousarray(src))
    gws = ((cols + 31) // 32 * 32, (rows + 31) // 32 * 32)
    _kern['transpose_f16'](_q, gws, (32, 32), sb, db, np.int32(rows), np.int32(cols))
    out = np.empty((cols, rows), dtype=np.float16)
    cl.enqueue_copy(_q, out, db)
    _q.finish()
    return out


def transpose_f32(src: np.ndarray) -> np.ndarray:
    init()
    rows, cols = src.shape
    n = rows * cols * 4
    sb = _buf(n, 0)
    db = _buf(n, 1)
    cl.enqueue_copy(_q, sb, np.ascontiguousarray(src))
    gws = ((cols + 31) // 32 * 32, (rows + 31) // 32 * 32)
    _kern['transpose_f32'](_q, gws, (32, 32), sb, db, np.int32(rows), np.int32(cols))
    out = np.empty((cols, rows), dtype=np.float32)
    cl.enqueue_copy(_q, out, db)
    _q.finish()
    return out


def hgemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """row-major C = a(M,K) @ b(K,N), fp16 in/out; returns fp16."""
    init()
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    ab = _buf(a.size * 2, 0)
    bb = _buf(b.size * 2, 1)
    cb = _buf(m * n * 2, 2)
    cl.enqueue_copy(_q, ab, np.ascontiguousarray(a))
    cl.enqueue_copy(_q, bb, np.ascontiguousarray(b))
    qp = ctypes.c_void_p(int(_q.int_ptr))
    rc = _bl.CLBlastHgemm(ROW, NO, NO, m, n, k, 0x3C00,
                          _ptr(ab), 0, a.shape[1],
                          _ptr(bb), 0, b.shape[1], 0x0000,
                          _ptr(cb), 0, n,
                          ctypes.byref(qp), None)
    if rc:
        raise RuntimeError('Hgemm rc=%d' % rc)
    out = np.empty((m, n), dtype=np.float16)
    cl.enqueue_copy(_q, out, cb)
    _q.finish()
    return out


def to_dev_f16(a: np.ndarray):
    init()
    b = _buf(a.size * 2, 50)
    cl.enqueue_copy(_q, b, np.ascontiguousarray(a))
    return b


def from_dev_f16(b, rows: int, cols: int) -> np.ndarray:
    init()
    out = np.empty((rows, cols), dtype=np.float16)
    cl.enqueue_copy(_q, out, b)
    _q.finish()
    return out


def hgemm_dev(a_dev, b_dev, m: int, n: int, k: int, b_trans: bool = False) -> np.ndarray:
    """A device (M,K) fp16, B device fp16 (K,N) or (N,K) if b_trans; returns host fp16 C."""
    init()
    cb = _buf(m * n * 2, 51)
    qp = ctypes.c_void_p(int(_q.int_ptr))
    b_ld = k if b_trans else n
    rc = _bl.CLBlastHgemm(ROW, NO, 112 if b_trans else NO, m, n, k, 0x3C00,
                          _ptr(a_dev), 0, k,
                          _ptr(b_dev), 0, b_ld, 0x0000,
                          _ptr(cb), 0, n,
                          ctypes.byref(qp), None)
    if rc:
        raise RuntimeError('Hgemm_dev rc=%d' % rc)
    return from_dev_f16(cb, m, n)


def hgemm_nt(a: np.ndarray, b_t: np.ndarray) -> np.ndarray:
    """row-major C = a(M,K) @ b_t(N,K)^T, fp16; b_t is the un-transposed (out,in) weight."""
    init()
    m, k = a.shape
    n, k2 = b_t.shape
    assert k == k2
    ab = _buf(a.size * 2, 0)
    bb = _buf(b_t.size * 2, 1)
    cb = _buf(m * n * 2, 2)
    cl.enqueue_copy(_q, ab, np.ascontiguousarray(a))
    cl.enqueue_copy(_q, bb, np.ascontiguousarray(b_t))
    qp = ctypes.c_void_p(int(_q.int_ptr))
    rc = _bl.CLBlastHgemm(ROW, NO, 112, m, n, k, 0x3C00,
                          _ptr(ab), 0, k,
                          _ptr(bb), 0, k, 0x0000,
                          _ptr(cb), 0, n,
                          ctypes.byref(qp), None)
    if rc:
        raise RuntimeError('Hgemm_nt rc=%d' % rc)
    out = np.empty((m, n), dtype=np.float16)
    cl.enqueue_copy(_q, out, cb)
    _q.finish()
    return out


def sgemm_nt(a: np.ndarray, b_t: np.ndarray) -> np.ndarray:
    init()
    m, k = a.shape
    n, k2 = b_t.shape
    assert k == k2
    ab = _buf(a.size * 4, 0)
    bb = _buf(b_t.size * 4, 1)
    cb = _buf(m * n * 4, 2)
    cl.enqueue_copy(_q, ab, np.ascontiguousarray(a))
    cl.enqueue_copy(_q, bb, np.ascontiguousarray(b_t))
    qp = ctypes.c_void_p(int(_q.int_ptr))
    rc = _bl.CLBlastSgemm(ROW, NO, 112, m, n, k, 1.0,
                          _ptr(ab), 0, k,
                          _ptr(bb), 0, k, 0.0,
                          _ptr(cb), 0, n,
                          ctypes.byref(qp), None)
    if rc:
        raise RuntimeError('Sgemm_nt rc=%d' % rc)
    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(_q, out, cb)
    _q.finish()
    return out


def sgemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """row-major C = a(M,K) @ b(K,N), fp32."""
    init()
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    ab = _buf(a.size * 4, 0)
    bb = _buf(b.size * 4, 1)
    cb = _buf(m * n * 4, 2)
    cl.enqueue_copy(_q, ab, np.ascontiguousarray(a))
    cl.enqueue_copy(_q, bb, np.ascontiguousarray(b))
    qp = ctypes.c_void_p(int(_q.int_ptr))
    rc = _bl.CLBlastSgemm(ROW, NO, NO, m, n, k, 1.0,
                          _ptr(ab), 0, a.shape[1],
                          _ptr(bb), 0, b.shape[1], 0.0,
                          _ptr(cb), 0, n,
                          ctypes.byref(qp), None)
    if rc:
        raise RuntimeError('Sgemm rc=%d' % rc)
    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(_q, out, cb)
    _q.finish()
    return out
