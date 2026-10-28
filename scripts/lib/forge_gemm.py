# -*- coding: utf-8 -*-
"""forge_gemm.py —— AVX2 GEMM 内核的 Python 封装（ctypes 调 forge_gemm.dll）。

为什么走 ctypes 而不是 torch C++ 扩展：
  扩展要拉 torch 头文件、走 MSVC 链接，Windows 上很脆；而 torch 张量的
  数据指针用 t.data_ptr() 就能拿到，传裸指针给 DLL 足够了。
  代价是只能吃【连续、float32、CPU】张量 —— 由这里的断言负责守住。

布局记号见 avx2_gemm_lib.cpp 文件头：
  I = C 的行数, J = C 的列数, R = 归约维
  NT(ta=0,tb=0): C = A[M,K] @ B[N,K]^T      <- Linear 前向 x @ W^T
  NN(ta=0,tb=1): C = A[M,K] @ B[K,N]        <- grad_x = grad_y @ W
  TN(ta=1,tb=1): C = A[K,M]^T @ B[K,N]      <- grad_W = grad_y^T @ x
  TT(ta=1,tb=0): C = A[K,M]^T @ B[N,K]^T
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import ctypes
import os

import torch

_DLL = os.environ.get('FORGE_GEMM_DLL',
                      r'D:\work\bitsandbytes-CPU\.autopilot\forge_gemm.dll')

_lib = ctypes.CDLL(_DLL)

_F = ctypes.c_float
_P = ctypes.POINTER(_F)

_lib.forge_set_threads.argtypes = [ctypes.c_int]
_lib.forge_set_threads.restype = None
_lib.forge_set_block.argtypes = [ctypes.c_int] * 4
_lib.forge_set_block.restype = None
_lib.forge_get_block.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4
_lib.forge_get_block.restype = None
_lib.forge_mm.argtypes = [_P, _P, _P,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.forge_mm.restype = ctypes.c_int
_lib.forge_set_auto.argtypes = [ctypes.c_int]
_lib.forge_set_auto.restype = None
_lib.forge_query_decomp.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 4
_lib.forge_query_decomp.restype = None
_lib.forge_info.restype = ctypes.c_char_p
_lib.forge_selfcheck.restype = ctypes.c_int


def set_auto(on):
    """on=1 按形状自适应推导 (NC,IC)；on=0 用 set_block 的固定值。"""
    _lib.forge_set_auto(1 if on else 0)


def query_decomp(I, J):
    """返回这个形状会被切成 (NC, njc, icc, nic) —— 任务数 = njc*nic。"""
    a, b, c, d = (ctypes.c_int() for _ in range(4))
    _lib.forge_query_decomp(int(I), int(J), a, b, c, d)
    return (a.value, b.value, c.value, d.value)


def set_threads(n):
    _lib.forge_set_threads(int(n))


def set_block(MC, NC, KC, IC):
    _lib.forge_set_block(int(MC), int(NC), int(KC), int(IC))


def get_block():
    # ★ 必须四个独立数组：ctypes 传同一个数组四次，C 那边会依次覆写同一块内存，
    #   读回来只剩最后一个值（之前就是读到 (512,0,0,0) 这种鬼东西）。
    a, b, c, d = (ctypes.c_int() for _ in range(4))
    _lib.forge_get_block(a, b, c, d)
    return (a.value, b.value, c.value, d.value)


def info():
    return _lib.forge_info().decode()


def selfcheck():
    return _lib.forge_selfcheck()


_LAYOUT = {'NT': (0, 0), 'NN': (0, 1), 'TN': (1, 1), 'TT': (1, 0)}


def _ptr(t):
    if not t.is_contiguous():
        raise ValueError('张量必须连续')
    if t.dtype != torch.float32:
        raise ValueError('只支持 float32，收到 %s' % t.dtype)
    if t.device.type != 'cpu':
        raise ValueError('只支持 CPU 张量，收到 %s' % t.device)
    return ctypes.cast(t.data_ptr(), _P)


def mm(A, B, layout='NT'):
    """按指定布局算矩阵乘。

    NT: A=[I,R] B=[J,R] -> [I,J]        NN: A=[I,R] B=[R,J] -> [I,J]
    TN: A=[R,I] B=[R,J] -> [I,J]        TT: A=[R,I] B=[J,R] -> [I,J]
    """
    ta, tb = _LAYOUT[layout]
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError('只支持 2D')
    if ta:
        R, I = A.shape
    else:
        I, R = A.shape
    if tb:
        R2, J = B.shape
    else:
        J, R2 = B.shape
    if R != R2:
        raise ValueError('归约维不一致: A 给 %d, B 给 %d' % (R, R2))
    C = torch.empty(I, J, dtype=torch.float32)
    rc = _lib.forge_mm(_ptr(A), _ptr(B), _ptr(C),
                       I, J, R, ta, tb, A.stride(0), B.stride(0), C.stride(0))
    if rc != 0:
        raise RuntimeError('forge_mm 返回错误码 %d' % rc)
    return C


def torch_ref(A, B, layout='NT'):
    """同一个布局下 torch 的等价写法，用于对照。"""
    if layout == 'NT':
        return A @ B.t()
    if layout == 'NN':
        return A @ B
    if layout == 'TN':
        return A.t() @ B
    if layout == 'TT':
        return A.t() @ B.t()
    raise ValueError(layout)


# Linear 的三种用法（形状约定跟 nn.Linear 一致：W 是 [out, in]）
def linear_fwd(x, W):
    """x=[..., in] W=[out, in] -> [..., out]"""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    y = mm(x2, W, 'NT')
    return y.reshape(*shp[:-1], W.shape[0])


def linear_bwd(grad_y, x, W):
    """返回 (grad_x, grad_W)。"""
    gy = grad_y.reshape(-1, grad_y.shape[-1])
    x2 = x.reshape(-1, x.shape[-1])
    gx = mm(gy, W, 'NN')          # [M,N] @ [N,K] -> [M,K]
    gW = mm(gy, x2, 'TN')         # [M,N]^T @ [M,K] -> [N,K]
    return gx.reshape(*x.shape), gW


if __name__ == '__main__':
    print('DLL : %s' % _DLL)
    print('info: %s' % info())
    print('block: %s' % (get_block(),))
    rc = selfcheck()
    print('selfcheck rc=%d' % rc)
