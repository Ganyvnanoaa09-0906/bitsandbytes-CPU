"""probe_opencl_conv.py — is im2col + CLBlast SGEMM faster than the CPU conv?

Measures representative SD1.5 UNet conv shapes. CPU baseline is the PyTorch conv on
6 threads; the candidate path is CPU unfold (im2col) + CLBlast SGEMM on the iGPU +
the fold back on CPU.  Everything is inference-only (no_grad).
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
LAYOUT_ROW = 101   # CLBlastLayoutRowMajor
TRANS_NO = 111     # CLBlastTransposeNo
TRANS_YES = 112


def setup():
    plats = cl.get_platforms()
    dev = plats[0].get_devices()[0]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
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
    return ctx, queue, bl




def gpu_sgemm(queue, bl, A, B):
    """C = A @ B on the iGPU; A (M,K), B (K,N) row-major; returns (M,N) numpy."""
    M, K = A.shape
    Kb, N = B.shape
    assert Kb == K
    a = np.ascontiguousarray(A, dtype=np.float32)
    b = np.ascontiguousarray(B, dtype=np.float32)
    mf = cl.mem_flags
    ctx = QUEUE.context
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    qp = ctypes.c_void_p(int(queue.int_ptr))
    rc = bl.CLBlastSgemm(
        LAYOUT_ROW, TRANS_NO, TRANS_NO, M, N, K,
        1.0,
        ctypes.c_void_p(int(a_buf.int_ptr)), 0, K,
        ctypes.c_void_p(int(b_buf.int_ptr)), 0, N,
        0.0,
        ctypes.c_void_p(int(c_buf.int_ptr)), 0, N,
        ctypes.byref(qp), None)
    assert rc == 0, rc
    out = np.empty((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, out, c_buf)
    queue.finish()
    a_buf.release()
    b_buf.release()
    c_buf.release()
    return out


def bench(shape, groups=1, iters=3):
    n, cin, h, w = shape
    cout = cin
    x = torch.randn(n, cin, h, w)
    conv = torch.nn.Conv2d(cin, cout, 3, padding=1, groups=groups).eval()
    with torch.no_grad():
        cpu_ref = conv(x)
        # CPU conv timing
        t0 = time.perf_counter()
        for _ in range(iters):
            y = conv(x)
        cpu_s = (time.perf_counter() - t0) / iters

        # candidate: unfold + GPU GEMM
        t0 = time.perf_counter()
        cols = F.unfold(x, 3, padding=1, stride=1)  # (n, cin*9, L)
        L = cols.shape[-1]
        cols = cols.reshape(n, groups, cin // groups * 9, L)
        wt = conv.weight.detach().reshape(groups, cout // groups, cin // groups * 9)
        t_unfold = time.perf_counter() - t0
        outs = []
        t0 = time.perf_counter()
        for g in range(groups):
            for i in range(n):
                o = gpu_sgemm(queue=QUEUE, bl=BL,
                              A=wt[g].numpy(),
                              B=cols[i, g].numpy())  # (cout/g, L)
                outs.append(torch.from_numpy(o))
        t_gpu = time.perf_counter() - t0
        if groups == 1:
            gpu_ref = torch.stack(outs).reshape(n, cout, h, w)
        else:
            gpu_ref = None
        err = None if gpu_ref is None else float((gpu_ref - cpu_ref).abs().max())
        gflops = 2 * cout * cin * 9 * n * h * w / 1e9
        print('shape=%s g=%d L=%d | cpu %.1f ms (%.0f GF) | unfold %.1f ms + gemm %.1f ms '
              '(%.0f GF gemm) | total %.1f ms | maxdiff %s'
              % (shape, groups, L, cpu_s * 1e3, gflops / cpu_s,
                 t_unfold * 1e3, t_gpu * 1e3, gflops / t_gpu,
                 (t_unfold + t_gpu) * 1e3, err))


if __name__ == '__main__':
    ctx, QUEUE, BL = setup()
    torch.set_num_threads(6)
    # representative SD1.5 UNet mid/high-res convs (batch 1)
    for shape in [(1, 320, 64, 64), (1, 640, 32, 32), (1, 1280, 16, 16),
                  (1, 320, 32, 32), (1, 640, 16, 16), (1, 1280, 8, 8)]:
        bench(shape)
