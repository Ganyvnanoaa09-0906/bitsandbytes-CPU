"""Does gfx902 have packed half2 instructions at all? Runtime-value kernel + ISA dump.

The pure-compute fp16 benchmark measured only 1.1x fp32, and the ISA of the constant-folded version
showed scalar `add_f16` on 32-bit registers. That is consistent with the compiler lowering half2 to two
scalar half ops. This probe uses data-dependent half2 values from global memory so the compiler cannot
constant-fold, then dumps the generated ISA and counts v_pk_* / v_mad_f16 / v_fma_f16 / v_dot*.
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

import glob
import os
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

SRC = r'''
__kernel void pack_probe(__global half* a_buf, __global half* b_buf,
                         __global float* out, const int iters) {
    int g = get_global_id(0);
    half2 a = vload2(0, a_buf + (g & 1023) * 2);
    half2 b = vload2(0, b_buf + (g & 1023) * 2);
    half2 c0 = a, c1 = b, c2 = a + b, c3 = a - b;
    for (int i = 0; i < iters; i++) {
        c0 = mad(a, c0, b);
        c1 = mad(b, c1, a);
        c2 = mad(a, c2, c1);
        c3 = mad(b, c3, c2);
    }
    float2 s = convert_float2(c0) + convert_float2(c1) + convert_float2(c2) + convert_float2(c3);
    out[g] = s.x + s.y;
}
'''
tmpdir = r'D:\work\bitsandbytes-CPU\_isa_dump2'
os.makedirs(tmpdir, exist_ok=True)
for old in glob.glob(tmpdir + r'\*'):
    try:
        os.remove(old)
    except OSError:
        pass
os.chdir(tmpdir)
prog = cl.Program(ctx, SRC).build(options=['-save-temps', '-cl-mad-enable', '-cl-fast-relaxed-math'])
print('build ok; ISA files:', [os.path.basename(p) for p in glob.glob(tmpdir + r'\*')])

tokens = ['v_pk_fma_f16', 'v_pk_mul_f16', 'v_pk_add_f16', 'v_pk_sub_f16', 'v_mad_f16',
          'v_fma_f16', 'v_add_f16', 'v_mul_f16', 'v_dot2', 'v_dot4', 'v_mac_f16']
for p in glob.glob(os.path.join(tmpdir, '*.isa')) + glob.glob(os.path.join(tmpdir, '*.s')):
    text = open(p, 'rb').read().decode('utf-8', errors='ignore')
    print()
    print('file:', os.path.basename(p))
    for t in tokens:
        n = text.count(t)
        if n:
            print('  %-12s x%d' % (t, n))
    # show the inner loop (lines around the first label with fmac)
    lines = text.split('\n')
    for i, ln in enumerate(lines):
        if 'f16' in ln and ('fma' in ln or 'mad' in ln or 'add' in ln or 'mul' in ln):
            print('  e.g.: %s' % ln.strip()[:90])
            break

# and measure it, for completeness
a = np.random.randn(2048).astype(np.float16)
b = np.random.randn(2048).astype(np.float16)
A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
N = 6 * 64 * 64
out = cl.Buffer(ctx, mf.WRITE_ONLY, N * 4)
k = cl.Kernel(prog, 'pack_probe')
iters = 20000
for _ in range(2):
    k(queue, (N,), (256,), A, B, out, np.int32(iters))
queue.finish()
t0 = time.perf_counter()
k(queue, (N,), (256,), A, B, out, np.int32(iters))
queue.finish()
dt = time.perf_counter() - t0
gf = N * 4 * iters * 4 / dt / 1e9
print()
print('pack_probe throughput: %.1f GFLOPS (4 half2 mads per iteration per work item)' % gf)
