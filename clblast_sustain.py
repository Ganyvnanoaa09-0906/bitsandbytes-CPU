"""clblast_sustain.py -- is CLBlast's 644.9 GFLOPS sustainable, or a burst?

The burst number is not the one that matters. This machine has a 15 W envelope
shared by CPU and iGPU, and the report already records that hammering a single
operator drives the clock down hardest: continuous CPU sgemm decayed 131.7 ->
106.9 GFLOPS over 90 seconds, while a real mixed training load measured 221-241
because it never saturates one unit that way.

bench_opencl_sustain.py asked exactly this question and wrote its criterion in
advance -- "if the iGPU holds 200+ sustained, against the CPU's sustained 107,
that is more than 2x and a custom backend is worth writing" -- but it used the
hand-written kernel's 245 GFLOPS, not CLBlast's 644.9. Same question, better
kernel, so it needs re-asking.

Method: run both backends for the same duration with the same shape, sampling
throughout, and report first value, last value, decay and sustained mean. The CPU
arm is measured in the same process and the same conditions, because the two
share a power budget and a comparison across sessions would be worthless.

Usage:
  python clblast_sustain.py --seconds 90 --n 2048 --sample 5
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'


def load_clblast():
    bl = ctypes.CDLL(DLL)
    bl.CLBlastSgemm.restype = ctypes.c_int
    # m, n, k are size_t in CLBlast, not int. Declaring them as c_int shifts every
    # later argument by four bytes, which surfaces as "argument 7: TypeError" on
    # alpha -- the symptom points one position past the actual mistake.
    bl.CLBlastSgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    return bl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=90.0)
    ap.add_argument('--n', type=int, default=2048)
    ap.add_argument('--sample', type=float, default=5.0)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()

    n = a.n
    flops = 2.0 * n ** 3
    print('持续性能测试: %d^3, 每个后端 %.0f 秒, 每 %.0f 秒采一次' % (n, a.seconds, a.sample))
    print()

    # ---------------- CLBlast ----------------
    import pyopencl as cl
    bl = load_clblast()
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    bufA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bufB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bufC = cl.Buffer(ctx, mf.WRITE_ONLY, A.nbytes)
    q.finish()

    # Values and wrapping copied from bench_clblast.py, which is known to work:
    # layout=1 is row-major and transpose=111 is "no transpose" (not 0, which is
    # invalid), and pyopencl buffers must be wrapped as c_void_p(int(buf.int_ptr))
    # rather than passed directly. Getting either wrong raises on a later argument
    # than the one that is actually incorrect.
    LAYOUT_ROW, TRANSPOSE_NO = 1, 111
    qp = ctypes.c_void_p(int(q.int_ptr))

    def one_gpu():
        rc = bl.CLBlastSgemm(LAYOUT_ROW, TRANSPOSE_NO, TRANSPOSE_NO, n, n, n,
                             ctypes.c_float(1.0),
                             ctypes.c_void_p(int(bufA.int_ptr)), 0, n,
                             ctypes.c_void_p(int(bufB.int_ptr)), 0, n,
                             ctypes.c_float(0.0),
                             ctypes.c_void_p(int(bufC.int_ptr)), 0, n,
                             ctypes.byref(qp), None)
        if rc != 0:
            raise RuntimeError('CLBlast 错误码 %d' % rc)
        q.finish()

    one_gpu()
    one_gpu()
    print('%-8s %12s %10s   %s' % ('后端', 'GFLOPS', '相对首值', '备注'))
    print('-' * 62)

    def run_arm(name, one, tag=''):
        first = None
        last = None
        vals = []
        t_end = time.perf_counter() + a.seconds
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            cnt = 0
            while time.perf_counter() - t0 < a.sample:
                one()
                cnt += 1
            dt = time.perf_counter() - t0
            gf = cnt * flops / dt / 1e9
            if first is None:
                first = gf
            last = gf
            vals.append(gf)
            print('%-8s %12.1f %9.1f%%   %s' % (name, gf, 100.0 * gf / first, tag))
            tag = ''
        mean_tail = sum(vals[len(vals) // 2:]) / max(1, len(vals) - len(vals) // 2)
        return first, last, mean_tail

    g_first, g_last, g_tail = run_arm('iGPU', one_gpu, 'CLBlast')
    print()

    # ---------------- CPU ----------------
    torch.set_num_threads(a.threads)
    x = torch.randn(n, n)
    y = torch.randn(n, n)
    x @ y

    def one_cpu():
        x @ y

    c_first, c_last, c_tail = run_arm('CPU', one_cpu, 'torch %d线程' % a.threads)
    print()

    print('=' * 62)
    print('%-10s %12s %12s %10s' % ('后端', '首值', '末值', '衰减'))
    print('%-10s %12.1f %12.1f %9.1f%%' % ('iGPU', g_first, g_last, 100 * (g_last / g_first - 1)))
    print('%-10s %12.1f %12.1f %9.1f%%' % ('CPU', c_first, c_last, 100 * (c_last / c_first - 1)))
    print()
    print('后半程均值: iGPU %.1f  CPU %.1f  ⇒ 【iGPU / CPU = %.2fx】'
          % (g_tail, c_tail, g_tail / c_tail))
    print()
    print('判据（bench_opencl_sustain.py 预先写明的）:')
    print('  核显持续保住 200+ 且相对 CPU 持续 2 倍以上 ⇒ 值得为它写自定义后端')
    print('  ⇒ 本次: iGPU 持续 %.1f, 相对 CPU %.2fx ⇒ %s'
          % (g_tail, g_tail / c_tail,
             '值得 ✓' if (g_tail >= 200 and g_tail / c_tail >= 2) else '不满足 ✗'))


if __name__ == '__main__':
    main()
