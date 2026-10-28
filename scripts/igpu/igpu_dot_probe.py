"""Two probes to find a path above 1127 GFLOPS on this iGPU.

1. Does the OpenCL compiler emit packed half2 math (v_pk_*)? Build a half2 kernel with -save-temps and
   grep the generated assembly for v_pk_fma / v_pk_mul. If it does and the throughput is still ~1.1x fp32,
   the hardware lacks the 2x rate and the fp16 route dead-ends at ~1100.
2. Does this compiler accept the GCN dot instructions (v_dot4_i32_i8 = 4 int8 MACs per instruction,
   i.e. a 4x fp32 ceiling) through __builtin_amdgcn_sdot4 or inline asm? That would be the only route
   with enough headroom to clear the GT 1030's 1127 GFLOPS by a wide margin.
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
print('device: %s | %d CU' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))

# ---------- probe 1: ISA ----------
HALF_KER = r'''
__kernel void h(__global float* out, const int iters) {
    int g = get_global_id(0);
    half2 a = (half2)(1.0001h, 1.0002h);
    half2 b = (half2)(0.9999h, 0.9998h);
    half2 c0 = a, c1 = a, c2 = a, c3 = a;
    for (int i = 0; i < iters; i++) {
        c0 = fma(a, c0, b); c1 = fma(a, c1, b); c2 = fma(a, c2, b); c3 = fma(a, c3, b);
    }
    float2 s = convert_float2(c0) + convert_float2(c1) + convert_float2(c2) + convert_float2(c3);
    out[g] = s.x + s.y;
}
'''
tmpdir = r'D:\work\bitsandbytes-CPU\_isa_dump'
os.makedirs(tmpdir, exist_ok=True)
for old in glob.glob(tmpdir + r'\*'):
    try:
        os.remove(old)
    except OSError:
        pass
os.chdir(tmpdir)
try:
    prog = cl.Program(ctx, HALF_KER).build(options=['-save-temps'])
    print('\n[1] build with -save-temps: OK')
except Exception as e:
    print('\n[1] build with -save-temps failed: %s' % str(e)[:200])

print('  files written:', [os.path.basename(p) for p in glob.glob(tmpdir + r'\*')][:20])
for ext in ('*.s', '*.asm', '*.ll', '*.bc', '*.spv'):
    for p in glob.glob(os.path.join(tmpdir, ext)):
        data = open(p, 'rb').read()
        text = data.decode('utf-8', errors='ignore')
        for token in ('v_pk_fma_f16', 'v_pk_mul_f16', 'v_fma_f16', 'v_mad_f16', 'v_pk_add_f16'):
            n = text.count(token)
            if n:
                print('  %s: %s x%d' % (os.path.basename(p), token, n))

# ---------- probe 2: dot instructions ----------
print('\n[2] GCN dot-instruction probes')
probes = {
    'builtin_sdot4': r'''
__kernel void d(__global float* out) {
    int a = (int)get_global_id(0);
    int b = a * 3;
    int acc = 0;
    acc = __builtin_amdgcn_sdot4(a, b, acc, 0);
    out[get_global_id(0)] = (float)acc;
}
''',
    'builtin_sdot8': r'''
__kernel void d(__global float* out) {
    int a = (int)get_global_id(0);
    int b = a * 3;
    int acc = 0;
    acc = __builtin_amdgcn_sdot8(a, b, acc, 0);
    out[get_global_id(0)] = (float)acc;
}
''',
    'inline_asm_dot4': r'''
__kernel void d(__global float* out) {
    int a = (int)get_global_id(0);
    int b = a * 3;
    int acc = 0;
    asm volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(acc) : "v"(a), "v"(b), "v"(acc));
    out[get_global_id(0)] = (float)acc;
}
''',
}
for name, src in probes.items():
    try:
        cl.Program(ctx, src).build()
        print('  %-18s COMPILES' % name)
    except Exception as e:
        msg = str(e)
        reason = 'unknown'
        for line in msg.split('\n'):
            if 'error:' in line:
                reason = line.strip()[:120]
                break
        print('  %-18s FAILS: %s' % (name, reason))
