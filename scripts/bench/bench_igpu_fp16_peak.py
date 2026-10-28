"""bench_igpu_fp16_peak.py -- is the iGPU's fp16 actually twice fp32?

The device reports cl_khr_fp16, so fp16 arithmetic is native and every GEMM written so
far has used fp32, leaving whatever fp16 offers untouched. AMD documents Vega's fp16
rate as twice fp32, but that figure normally refers to packed operations -- two halves
per register, accumulated in fp16. fp16 storage with fp32 accumulation runs at the same
rate as fp32. So the useful question is not "does fp16 help" but "which form helps, and
what does it cost in precision", and only measurement settles it.

Three kernels, all compute-bound with no memory traffic, each with eight independent
FMA chains so this measures throughput rather than latency -- a single dependent chain
would report the pipeline depth instead:

  fp32            float  fma, 1 flop per lane per issue
  fp16_fp32acc    half storage, accumulated in float
  fp16_packed     half2 pairs, accumulated in half

Reported as GFLOPS against the 1150 fp32 peak the existing benchmarks cite. If
fp16_packed is not near 2x fp32, the documented figure does not apply here and the
LDS-layout work stays the more promising direction.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

import pyopencl as cl  # noqa: E402

SRC = r'''
#define CHAINS 8
#define ITERS  4096

__kernel void peak_fp32(__global float* out) {
    const int gid = get_global_id(0);
    float a[CHAINS], b[CHAINS];
    #pragma unroll
    for (int i = 0; i < CHAINS; i++) { a[i] = (float)(gid + i) * 1e-3f; b[i] = 0.999f; }
    for (int it = 0; it < ITERS; it++) {
        #pragma unroll
        for (int i = 0; i < CHAINS; i++) {
            a[i] = fma(a[i], b[i], 0.5f);
            b[i] = fma(b[i], a[i], 0.25f);
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < CHAINS; i++) s += a[i] + b[i];
    out[gid] = s;
}

__kernel void peak_fp16_fp32acc(__global float* out) {
    const int gid = get_global_id(0);
    float a[CHAINS], b[CHAINS];
    #pragma unroll
    for (int i = 0; i < CHAINS; i++) { a[i] = (float)(gid + i) * 1e-3f; b[i] = 0.999f; }
    half ha[CHAINS], hb[CHAINS];
    #pragma unroll
    for (int i = 0; i < CHAINS; i++) { ha[i] = (half)a[i]; hb[i] = (half)b[i]; }
    for (int it = 0; it < ITERS; it++) {
        #pragma unroll
        for (int i = 0; i < CHAINS; i++) {
            a[i] = fma((float)ha[i], (float)hb[i], a[i]);
            b[i] = fma((float)hb[i], (float)ha[i], b[i]);
            ha[i] = (half)a[i];
            hb[i] = (half)b[i];
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < CHAINS; i++) s += a[i] + b[i];
    out[gid] = s;
}

__kernel void peak_fp16_packed(__global float* out) {
    const int gid = get_global_id(0);
    // two independent halves per register, accumulated in half -- the form that is
    // documented as 2x, and the one whose precision has to be checked separately
    half2 a[CHAINS], b[CHAINS];
    #pragma unroll
    for (int i = 0; i < CHAINS; i++) {
        a[i] = (half2)((half)(gid * 1e-4f), (half)(0.5f));
        b[i] = (half2)((half)0.999f, (half)0.998f);
    }
    for (int it = 0; it < ITERS; it++) {
        // the accumulator must depend on the iteration, or the compiler folds the
        // whole loop away -- the first version of this kernel reported 12369 GFLOPS,
        // ten times the device peak, because it accumulated into a constant
        half2 c = (half2)((half)(0.5f + it * 1e-6f), (half)(0.25f + it * 5e-7f));
        #pragma unroll
        for (int i = 0; i < CHAINS; i++) {
            a[i] = fma(a[i], b[i], c);
            b[i] = fma(b[i], a[i], c);
        }
    }
    half2 s = (half2)(0.f, 0.f);
    #pragma unroll
    for (int i = 0; i < CHAINS; i++) s += a[i] + b[i];
    out[gid] = (float)(s.s0 + s.s1);
}
'''

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
try:
    prg = cl.Program(ctx, SRC).build()
except Exception as e:
    print('  编译失败:')
    print(str(e)[:2000])
    raise SystemExit(1)

G = 6 * 256 * 32          # enough work items to fill 6 CUs several times over
ITERS, CHAINS = 4096, 8
buf = cl.Buffer(ctx, mf.WRITE_ONLY, G * 4)
print('设备 %s | %d CU | 工作项 %d' % (dev.name, dev.max_compute_units, G))
print()
print('%-20s %10s %12s %10s  %s' % ('内核', '秒', 'GFLOPS', '相对 fp32', '说明'))
print('-' * 74)

base = None
for name, flops_per_lane in (('peak_fp32', 2.0), ('peak_fp16_fp32acc', 2.0),
                             ('peak_fp16_packed', 4.0)):
    try:
        k = cl.Kernel(prg, name)
        k(queue, (G,), None, buf)
        queue.finish()
        t0 = time.perf_counter()
        for _ in range(3):
            k(queue, (G,), None, buf)
        queue.finish()
        dt = (time.perf_counter() - t0) / 3
        # each lane: CHAINS chains x 2 fma x ITERS iterations x flops_per_lane
        gf = G * CHAINS * 2.0 * ITERS * flops_per_lane / dt / 1e9
        if base is None:
            base = gf
        note = ''
        if 'fp16_packed' in name:
            note = '2 半/寄存器，fp16 累加'
        elif 'fp16' in name:
            note = 'fp16 存、fp32 累加'
        print('%-20s %10.4f %12.1f %9.2fx  %s' % (name, dt, gf, gf / base, note))
    except Exception as e:
        print('%-20s %10s %12s %10s  %s' % (name, 'FAIL', '-', '-', str(e)[:34]))
print()
print('判据: packed fp16 若接近 2x fp32 ⇒ AMD 的文档数字在本机成立；')
print('      若 ≈1x ⇒ 那个 2x 不适用于本机，应继续 LDS 布局方向（7.19/7.22）。')
