# -*- coding: utf-8 -*-
"""wire_fused_v2.py — 把 v2 融合核（strided SwiGLU + in-place add_scale）接进 bnb"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import io, os

T = r'D:\work\bitsandbytes-CPU\bitsandbytes'
BLOCK = r'D:\work\bitsandbytes-CPU\fused_kernels_v2.cpp'


def read(p):
    return io.open(p, encoding='utf-8').read()


def write(p, s):
    s = s.replace('\r\n', '\n').replace('\n', '\r\n')
    io.open(p, 'w', encoding='utf-8', newline='').write(s)


# ---- 1. cpu_ops.cpp ----
p = os.path.join(T, 'csrc', 'cpu_ops.cpp')
src = read(p)
if 'fused_swiglu_strided_fwd_cpu' in src:
    print('1. cpu_ops.cpp   v2 已接入')
else:
    blk = read(BLOCK)
    src = src.rstrip() + '\r\n\r\n' + blk.strip() + '\r\n'
    write(p, src)
    print('1. cpu_ops.cpp   +%d 字符' % len(blk))

# ---- 2. cpu_ops.h ----
DECL = '''
// ---- fused v2: strided SwiGLU + in-place add_scale + 融合链 (2026-09-18) ----
#if defined(__cplusplus)
extern "C" {
#endif
long long fused_swiglu_strided_fwd_cpu(
    const float* gate, long long gs0, long long gs1,
    const float* up, long long us0, long long us1,
    float* out, long long os0, long long M, long long D
);
long long fused_swiglu_strided_bwd_cpu(
    const float* gate, long long gs0, long long gs1,
    const float* up, long long us0, long long us1,
    const float* dout, long long ds0, long long ds1,
    float* dgate, long long dg0, long long dg1,
    float* dup, long long du0, long long du1,
    long long M, long long D
);
long long fused_add_scale_inplace_cpu(
    float* x, const float* y, const float* scale, long long M, long long D
);
long long fused_add_scale_rmsnorm_cpu(
    float* x, const float* y, const float* scale,
    const float* weight, float* out, float* xs, long long M, long long D, float eps
);
#if defined(__cplusplus)
}
#endif
'''
p = os.path.join(T, 'csrc', 'cpu_ops.h')
src = read(p)
if 'fused_swiglu_strided_fwd_cpu' in src:
    print('2. cpu_ops.h     v2 已接入')
else:
    idx = src.rstrip().rfind('#endif')
    src = src[:idx] + DECL + '\r\n' + src[idx:]
    write(p, src)
    print('2. cpu_ops.h     +%d 字符' % len(DECL))

# ---- 3. pythonInterface.cpp ----
BIND = '''
// ---- fused v2 (2026-09-18) ----
long long cfused_swiglu_strided_fwd_cpu(
    const float* gate, long long gs0, long long gs1,
    const float* up, long long us0, long long us1,
    float* out, long long os0, long long M, long long D
) {
    return fused_swiglu_strided_fwd_cpu(gate, gs0, gs1, up, us0, us1, out, os0, M, D);
}

long long cfused_swiglu_strided_bwd_cpu(
    const float* gate, long long gs0, long long gs1,
    const float* up, long long us0, long long us1,
    const float* dout, long long ds0, long long ds1,
    float* dgate, long long dg0, long long dg1,
    float* dup, long long du0, long long du1,
    long long M, long long D
) {
    return fused_swiglu_strided_bwd_cpu(gate, gs0, gs1, up, us0, us1, dout, ds0, ds1,
                                        dgate, dg0, dg1, dup, du0, du1, M, D);
}

long long cfused_add_scale_inplace_cpu(
    float* x, const float* y, const float* scale, long long M, long long D
) {
    return fused_add_scale_inplace_cpu(x, y, scale, M, D);
}

long long cfused_add_scale_rmsnorm_cpu(
    float* x, const float* y, const float* scale,
    const float* weight, float* out, float* xs, long long M, long long D, float eps
) {
    return fused_add_scale_rmsnorm_cpu(x, y, scale, weight, out, xs, M, D, eps);
}
'''
p = os.path.join(T, 'csrc', 'pythonInterface.cpp')
src = read(p)
if 'cfused_swiglu_strided_fwd_cpu' in src:
    print('3. pythonInterface v2 已接入')
else:
    lines = src.split('\n')
    last = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '}':
            last = i
            break
    lines.insert(last, BIND)
    write(p, '\n'.join(lines))
    print('3. pythonInterface +%d 字符' % len(BIND))

# ---- 4. export.def ----
EXPORTS = '''    cfused_swiglu_strided_fwd_cpu
    cfused_swiglu_strided_bwd_cpu
    cfused_add_scale_inplace_cpu
    cfused_add_scale_rmsnorm_cpu
'''
p = os.path.join(T, 'build_manual', 'export.def')
src = read(p)
if 'cfused_swiglu_strided_fwd_cpu' in src:
    print('4. export.def    v2 已接入')
else:
    src = src.rstrip() + '\n' + EXPORTS
    write(p, src)
    print('4. export.def    +4 个导出')

print()
print('v2 接入完成。')
