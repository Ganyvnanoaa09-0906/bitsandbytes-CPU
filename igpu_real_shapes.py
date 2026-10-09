"""igpu_real_shapes.py -- benchmark the GEMMs this model actually issues.

The square-matrix numbers are misleading in both directions. CLBlast at 2048^3
reaches 644.9 GFLOPS (3.21x the CPU) but at 512^3 only 94.4 (slower than the CPU),
so "is the iGPU worth it" cannot be answered from a square sweep. It depends
entirely on the shapes the model emits.

And it is not obvious which way that cuts. The model is d_model=512 with
ffn_dim=2048, which sounds small -- but at training time M is the TOKEN COUNT
(1024), not the batch size, so the per-layer GEMMs are (1024x512)@(512x2736) and
similar. That is ~2.9 GFLOP per call, firmly in the range where CLBlast was fast.
Equally, extraction reads the shapes off the model rather than assuming them.

Transfer cost is included, because on an APU it is a shared-memory copy rather
than a PCIe round trip and that is exactly what makes offload plausible here at
all. A shape only counts as a win if it wins after the copies.

Usage:  python igpu_real_shapes.py [--tokens 1024] [--batch 8]
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'


def load_clblast():
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
    return bl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=1024)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--reps', type=int, default=5)
    a = ap.parse_args()

    import pyopencl as cl
    bl = load_clblast()
    ctx = cl.create_some_context(interactive=False)
    dev = ctx.devices[0]
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    LAYOUT_ROW, TRANSPOSE_NO = 1, 111
    qp = ctypes.c_void_p(int(queue.int_ptr))
    torch.set_num_threads(a.threads)

    T = a.tokens
    print('设备 %s | %d CU | %d MB' % (dev.name, dev.max_compute_units,
                                       dev.global_mem_size // 2**20))
    print('M = token 数 = %d, torch %d 线程' % (T, a.threads))
    print()

    # (label, M, N, K) -- shapes the model issues at training time
    shapes = [
        ('qkv 投影',      T, 1024, 512),
        ('attn 分数 QK^T', T, T,    512),
        ('attn 加权 PV',   T, 512,  T),
        ('输出投影',       T, 512,  512),
        ('MLP w12',       T, 2736, 512),
        ('MLP w3',        T, 512,  2736),
        ('词表头',         T, 1024, 512),
    ]

    def gpu_gemm(m, n, k, reps, with_transfer):
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        t0 = time.perf_counter()
        ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        bc = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * 4)
        out = np.empty((m, n), dtype=np.float32)
        transfer = time.perf_counter() - t0
        args = (LAYOUT_ROW, TRANSPOSE_NO, TRANSPOSE_NO, m, n, k, ctypes.c_float(1.0),
                ctypes.c_void_p(int(ba.int_ptr)), 0, k,
                ctypes.c_void_p(int(bb.int_ptr)), 0, n,
                ctypes.c_float(0.0),
                ctypes.c_void_p(int(bc.int_ptr)), 0, n,
                ctypes.byref(qp), None)
        rc = bl.CLBlastSgemm(*args)
        if rc != 0:
            raise RuntimeError('CLBlast 错误码 %d' % rc)
        queue.finish()
        t0 = time.perf_counter()
        for _ in range(reps):
            rc = bl.CLBlastSgemm(*args)
            if rc != 0:
                raise RuntimeError('CLBlast 错误码 %d' % rc)
            queue.finish()
        dt = (time.perf_counter() - t0) / reps
        if with_transfer:
            cl.enqueue_copy(queue, out, bc).wait()
            dt += transfer + (time.perf_counter() - t0 - dt * reps) / max(reps, 1)
        return dt

    def cpu_gemm(m, n, k, reps):
        x = torch.randn(m, k)
        y = torch.randn(k, n)
        x @ y
        t0 = time.perf_counter()
        for _ in range(reps):
            x @ y
        return (time.perf_counter() - t0) / reps

    print('%-16s %-22s %10s %10s %8s  %s'
          % ('层', '形状 MxNxK', 'CPU s', 'iGPU s', '核显/CPU', '结论'))
    print('-' * 88)
    tot_c = tot_g = tot_flops = 0.0
    for label, m, n, k in shapes:
        fl = 2.0 * m * n * k
        try:
            tc = cpu_gemm(m, n, k, a.reps)
            tg = gpu_gemm(m, n, k, a.reps, with_transfer=True)
            ratio = tg / tc
            verdict = '核显快 %.2fx ✓' % (1 / ratio) if ratio < 1 else 'CPU 快 %.2fx' % ratio
            print('%-16s %-22s %10.5f %10.5f %7.2fx  %s'
                  % (label, '%dx%dx%d' % (m, n, k), tc, tg, ratio, verdict))
            tot_c += tc
            tot_g += tg
            tot_flops += fl
        except Exception as e:
            print('%-16s %-22s %10s %10s %8s  %s: %s'
                  % (label, '%dx%dx%d' % (m, n, k), 'FAIL', '-', '-',
                     type(e).__name__, str(e)[:28]))
    print()
    print('单层合计（%d 次调用）: CPU %.4f s   iGPU %.4f s（含传输）' % (len(shapes), tot_c, tot_g))
    if tot_g > 0:
        print('  ⇒ 核显/CPU = %.2fx  ⇒ %s'
              % (tot_g / tot_c, '核显更优 ✓' if tot_g < tot_c else 'CPU 更优 ✗'))
        print('  说明：全模型 %d 层，逐层按此比例放大；M=%d 已含 token 维。' % (10, T))


if __name__ == '__main__':
    main()
