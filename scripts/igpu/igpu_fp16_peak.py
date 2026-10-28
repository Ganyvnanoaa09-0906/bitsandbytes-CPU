"""What is this iGPU's real FP16 throughput? Everything below depends on it.

The report says Vega 6 peak FP32 = 1150 GFLOPS and measured pure-FMA = 955-1089 (83-95%). Vega is GCN 5,
the first AMD architecture with "rapid packed math": packed half2 operations at twice the FP32 rate, so
the FP16 ceiling should be about 2300 GFLOPS. A previous note in this repo claims 4665 GFLOPS for packed
FP16 (4.28x FP32), which is twice that ceiling and therefore either a different FLOP count or a
measurement error -- per the report's own rule, anything above the rated peak gets treated as a
measurement error until proven otherwise. So measure both peaks with the same harness in one session:

  fp32: 8 independent accumulators, fma(a, x, b)
  fp16: 8 independent half2 accumulators, fma(a2, x2, b2) -- 2 FLOPs per lane per op

Also prints the device's max clock and the cl_khr_fp16 extension.
"""
from __future__ import annotations
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

import sys
import time

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
print('device: %s | %d CU | max clock %d MHz' % (dev.name, dev.max_compute_units,
                                                 getattr(dev, 'max_clock_frequency', 0) // 1000000))
print('extensions with fp16:', [e for e in dev.extensions.split() if 'fp16' in e])

SRC = r'''
__kernel void peak_f32(__global float* out, const int iters) {
    float a = 1.000001f, b = 0.999999f;
    float c0=a, c1=a, c2=a, c3=a, c4=a, c5=a, c6=a, c7=a;
    for (int i = 0; i < iters; i++) {
        c0 = fma(a, c0, b); c1 = fma(a, c1, b); c2 = fma(a, c2, b); c3 = fma(a, c3, b);
        c4 = fma(a, c4, b); c5 = fma(a, c5, b); c6 = fma(a, c6, b); c7 = fma(a, c7, b);
    }
    out[get_global_id(0)] = c0+c1+c2+c3+c4+c5+c6+c7;
}

__kernel void peak_f16(__global float* out, const int iters) {
    half2 a = (half2)(1.0001h, 1.0002h);
    half2 b = (half2)(0.9999h, 0.9998h);
    half2 c0=a, c1=a, c2=a, c3=a, c4=a, c5=a, c6=a, c7=a;
    for (int i = 0; i < iters; i++) {
        c0 = fma(a, c0, b); c1 = fma(a, c1, b); c2 = fma(a, c2, b); c3 = fma(a, c3, b);
        c4 = fma(a, c4, b); c5 = fma(a, c5, b); c6 = fma(a, c6, b); c7 = fma(a, c7, b);
    }
    float2 s = convert_float2(c0) + convert_float2(c1) + convert_float2(c2) + convert_float2(c3)
             + convert_float2(c4) + convert_float2(c5) + convert_float2(c6) + convert_float2(c7);
    out[get_global_id(0)] = s.x + s.y;
}
'''
prog = cl.Program(ctx, SRC).build()

LANES = 6 * 64
WGS = 256
N = LANES * 64          # plenty of work items; each CU gets many waves
iters = 20000
nbytes = N * 4
out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, nbytes)


def bench(kern, ops_per_lane):
    k = cl.Kernel(prog, kern)
    for _ in range(2):
        k(queue, (N,), (WGS,), out, np.int32(iters))
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(3):
        k(queue, (N,), (WGS,), out, np.int32(iters))
    queue.finish()
    dt = (time.perf_counter() - t0) / 3.0
    # each thread: 8 accumulators x iters x 2 FLOPs (fp32) or 4 FLOPs (half2)
    flops = N * 8 * iters * ops_per_lane
    return flops / dt / 1e9


print()
print('work items %d x %d iters x 8 accumulators' % (N, iters))
g32 = bench('peak_f32', 2)
g16 = bench('peak_f16', 4)
print('  fp32 peak measured: %8.1f GFLOPS  (%.1f%% of 1150)' % (g32, 100 * g32 / 1150))
print('  fp16 peak measured: %8.1f GFLOPS  (%.1f%% of 2300)' % (g16, 100 * g16 / 2300))
print('  fp16/fp32 ratio   : %8.2fx' % (g16 / g32))
print()
print('  the GT 1030 FP32 theoretical value is 1127 GFLOPS')
print('  fp16/fp32 headroom at the measured peaks: %.2fx' % (g16 / g32))
