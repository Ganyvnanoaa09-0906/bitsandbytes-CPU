"""Push the fp16 pure-compute number from ~1100 to >= 1127 (the GT 1030's theoretical FP32 value).

The measured ceiling on this iGPU is fp32-like: the compiler lowers half2 mad to scalar f16 ops (ISA
shows v_pk_add_f16 present but scalar mad), so the FLOP rate is bounded by the same 2 FLOPs/lane/cycle as
fp32 -- theoretical 1150 GFLOPS at 1.5 GHz. The best measurement so far is 1099 (95.6% of that). Crossing
1127 therefore needs >98% issue efficiency, which means removing every loop and scheduling overhead:
more independent accumulators, fully unrolled inner blocks, and the right work size.

Sweeps: accumulators in {8, 12, 16, 24, 32} x work-size in {24576, 49152, 98304} x unroll in {1, 2, 4}.
Reports the best and whether it crosses 1127.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
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
iface = cl.mem_flags

SRC_TMPL = r'''
__kernel void mk(__global float* out, const int iters) {
    int g = get_global_id(0);
    half2 a = (half2)(1.0001h + 1e-5h * (g & 7), 1.0002h - 1e-5h * (g & 7));
    half2 b = (half2)(0.9999h - 1e-5h * ((g >> 3) & 7), 0.9998h + 1e-5h * ((g >> 3) & 7));
    half2 c[%(NACC)d];
    #pragma unroll
    for (int j = 0; j < %(NACC)d; j++) c[j] = a * (1.0h + 0.01h * (j + 1));
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int u = 0; u < %(UNROLL)d; u++) {
            #pragma unroll
            for (int j = 0; j < %(NACC)d; j++) c[j] = mad(a, c[j], b);
        }
    }
    float2 s = (float2)(0.0f, 0.0f);
    #pragma unroll
    for (int j = 0; j < %(NACC)d; j++) s += convert_float2(c[j]);
    out[g] = s.x + s.y;
}
'''

print('device: %s | %d CU' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))
print()
print('  %-6s %-8s %-6s %10s %10s' % ('nacc', 'worksize', 'unroll', 'GFLOPS', '% of 1127'))
best = (0.0, None)
for nacc in (8, 12, 16, 24, 32):
    for unroll in (1, 2, 4):
        src = SRC_TMPL % {'NACC': nacc, 'UNROLL': unroll}
        try:
            prog = cl.Program(ctx, src).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
            k = cl.Kernel(prog, 'mk')
        except Exception as e:
            print('  build failed nacc=%d unroll=%d: %s' % (nacc, unroll, str(e)[:80]))
            continue
        for size in (6 * 64 * 64, 6 * 64 * 128, 6 * 64 * 256):
            out = cl.Buffer(ctx, iface.WRITE_ONLY, size * 4)
            iters = 4000
            for _ in range(2):
                k(queue, (size,), (256,), out, np.int32(iters))
            queue.finish()
            t0 = time.perf_counter()
            k(queue, (size,), (256,), out, np.int32(iters))
            queue.finish()
            dt = time.perf_counter() - t0
            gf = size * nacc * unroll * iters * 4 / dt / 1e9
            if gf > best[0]:
                best = (gf, (nacc, size, unroll))
            if gf > 1050:
                print('  %-6d %-8d %-6d %10.1f %9.1f%%' % (nacc, size, unroll, gf, 100 * gf / 1127))
            del out

print()
print('  best: %.1f GFLOPS with nacc=%d worksize=%d unroll=%d'
      % (best[0], best[1][0], best[1][1], best[1][2]))
print('  target 1127 -> %s' % ('REACHED' if best[0] >= 1127 else 'still %.1f%% short' % (100 * (1 - best[0] / 1127))))
