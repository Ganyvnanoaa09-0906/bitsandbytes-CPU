"""iGPU fp16 pure-compute throughput measured as work done over a fixed 10 s window.

Used to run the iGPU side of the machine-level test: the same 10 s window in which
cpu_fma_full.exe is being looped from the shell. Prints one line: FLOPS achieved.
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
SRC = r'''
__kernel void peak_f16(__global float* out, const int iters) {
    int g = get_global_id(0);
    half2 a = (half2)(1.0001h + 1e-5h * (g & 7), 1.0002h - 1e-5h * (g & 7));
    half2 b = (half2)(0.9999h - 1e-5h * ((g >> 3) & 7), 0.9998h + 1e-5h * ((g >> 3) & 7));
    half2 c0 = a*1.1h, c1 = a*1.2h, c2 = a*1.3h, c3 = a*1.4h;
    half2 c4 = a*1.5h, c5 = a*1.6h, c6 = a*1.7h, c7 = a*1.8h;
    for (int i = 0; i < iters; i++) {
        c0 = mad(a,c0,b); c1 = mad(a,c1,b); c2 = mad(a,c2,b); c3 = mad(a,c3,b);
        c4 = mad(a,c4,b); c5 = mad(a,c5,b); c6 = mad(a,c6,b); c7 = mad(a,c7,b);
    }
    float2 s = convert_float2(c0)+convert_float2(c1)+convert_float2(c2)+convert_float2(c3)
             + convert_float2(c4)+convert_float2(c5)+convert_float2(c6)+convert_float2(c7);
    out[g] = s.x + s.y;
}
'''
prog = cl.Program(ctx, SRC).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
k = cl.Kernel(prog, 'peak_f16')
N = 6 * 64 * 64
iters = 20000
out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, N * 4)
for _ in range(2):
    k(queue, (N,), (256,), out, np.int32(iters))
queue.finish()

dur = 10.0
n = 0
t0 = time.perf_counter()
while time.perf_counter() - t0 < dur:
    k(queue, (N,), (256,), out, np.int32(iters))
    queue.finish()
    n += 1
el = time.perf_counter() - t0
gf = n * N * 8 * iters * 4 / el / 1e9
print('IGPU_FP16_GFLOPS %.1f  (runs %d, %.2f s)' % (gf, n, el))
