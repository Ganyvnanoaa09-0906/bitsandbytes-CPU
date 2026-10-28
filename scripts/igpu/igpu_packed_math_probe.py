"""Can this compiler emit packed half2 FMA (v_pk_fma_f16) at all?

The corrected fp16 pure-compute benchmark measured 1029 GFLOPS, only 1.09x the fp32 figure (945). That is
the signature of *scalar* half math: the compiler is not using the packed instruction, so the two lanes
are executed one after the other. Vega's packed-math peak is ~2x fp32, so if the compiler can be pushed
into emitting it, pure fp16 should land near 1900 GFLOPS, comfortably above the GT 1030's 1127.

Variants tried on the same loop (8 independent accumulators, data-dependent constants):
  A. fma(half2, half2, half2)                     -- measured before: 1029
  B. mad(half2, half2, half2)
  C. c = a * c + b  (separate mul and add)
  D. B/C with the build options -cl-mad-enable -cl-fast-relaxed-math
  E. float baseline for the same loop shape
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import sys
import time

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
print('device: %s | %d CU' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))

HEAD = r'''
#define INIT \
    int g = get_global_id(0); \
    half2 a = (half2)(1.0001h + 1e-5h * (g & 7), 1.0002h - 1e-5h * (g & 7)); \
    half2 b = (half2)(0.9999h - 1e-5h * ((g >> 3) & 7), 0.9998h + 1e-5h * ((g >> 3) & 7)); \
    half2 c0 = a * 1.1h, c1 = a * 1.2h, c2 = a * 1.3h, c3 = a * 1.4h; \
    half2 c4 = a * 1.5h, c5 = a * 1.6h, c6 = a * 1.7h, c7 = a * 1.8h;
#define BODY(EXPR) \
    for (int i = 0; i < iters; i++) { \
        c0 = EXPR(a, c0, b); c1 = EXPR(a, c1, b); c2 = EXPR(a, c2, b); c3 = EXPR(a, c3, b); \
        c4 = EXPR(a, c4, b); c5 = EXPR(a, c5, b); c6 = EXPR(a, c6, b); c7 = EXPR(a, c7, b); \
    }
#define TAIL(NAME) \
    float2 s = convert_float2(c0)+convert_float2(c1)+convert_float2(c2)+convert_float2(c3) \
             + convert_float2(c4)+convert_float2(c5)+convert_float2(c6)+convert_float2(c7); \
    out[g] = s.x + s.y;
'''

SRC = HEAD + r'''
#define FMA(a,c,b) fma(a,c,b)
#define MAD(a,c,b) mad(a,c,b)
#define MULADD(a,c,b) (a*c+b)

__kernel void v_fma(__global float* out, const int iters) { INIT BODY(FMA) TAIL(v_fma) }
__kernel void v_mad(__global float* out, const int iters) { INIT BODY(MAD) TAIL(v_mad) }
__kernel void v_muladd(__global float* out, const int iters) { INIT BODY(MULADD) TAIL(v_muladd) }
__kernel void v_f32(__global float* out, const int iters) {
    int g = get_global_id(0);
    float a = 1.000001f + 1e-7f * (g & 7);
    float b = 0.999999f - 1e-7f * ((g >> 3) & 7);
    float c0 = a*1.1f, c1 = a*1.2f, c2 = a*1.3f, c3 = a*1.4f;
    float c4 = a*1.5f, c5 = a*1.6f, c6 = a*1.7f, c7 = a*1.8f;
    for (int i = 0; i < iters; i++) {
        c0 = fma(a,c0,b); c1 = fma(a,c1,b); c2 = fma(a,c2,b); c3 = fma(a,c3,b);
        c4 = fma(a,c4,b); c5 = fma(a,c5,b); c6 = fma(a,c6,b); c7 = fma(a,c7,b);
    }
    out[g] = c0+c1+c2+c3+c4+c5+c6+c7;
}
'''

LANES = 6 * 64
N = LANES * 64
iters = 20000
out = cl.Buffer(ctx, mf.WRITE_ONLY, N * 4)


def measure(prog, name, ops=4):
    try:
        k = cl.Kernel(prog, name)
        for _ in range(2):
            k(queue, (N,), (256,), out, np.int32(iters))
        queue.finish()
        t0 = time.perf_counter()
        for _ in range(3):
            k(queue, (N,), (256,), out, np.int32(iters))
        queue.finish()
        dt = (time.perf_counter() - t0) / 3.0
        gf = N * 8 * iters * ops / dt / 1e9
        print('  %-12s %8.1f GFLOPS' % (name, gf))
        return gf
    except Exception as e:
        print('  %-12s FAILED: %s' % (name, str(e)[:80]))
        return 0.0


print()
print('=== build variants ===')
for opts in ('', '-cl-mad-enable -cl-fast-relaxed-math'):
    print('  options: %r' % (opts or '(none)'))
    try:
        prog = cl.Program(ctx, SRC).build(opts.split() if opts else [])
    except Exception as e:
        print('    build failed: %s' % str(e)[:200])
        continue
    measure(prog, 'v_f32', ops=2)
    measure(prog, 'v_fma', ops=4)
    measure(prog, 'v_mad', ops=4)
    measure(prog, 'v_muladd', ops=4)

print()
print('  reference: fp32 rated peak 1150 · GT 1030 fp32 theoretical 1127')
