# -*- coding: utf-8 -*-
"""wire_fused_kernels.py — 把融合核接入 bnb 的四个层次

1. csrc/cpu_ops.cpp      —— 追加融合核实现
2. csrc/cpu_ops.h        —— 追加声明
3. csrc/pythonInterface.cpp —— 追加 C 入口
4. build_manual/export.def  —— 追加导出

幂等：已接入过就跳过。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import io
import os

T = r'D:\work\bitsandbytes-CPU\bitsandbytes'
BLOCK = r'D:\work\bitsandbytes-CPU\fused_kernels_block.cpp'

MARK = 'fused_rmsnorm_fwd_cpu'   # 标记：已存在就跳过


def read(p):
    return io.open(p, encoding='utf-8').read()


def write(p, s):
    # 保持 CRLF（MSVC + GBK 代码页下 LF 中文注释会出问题）
    s = s.replace('\r\n', '\n').replace('\n', '\r\n')
    io.open(p, 'w', encoding='utf-8', newline='').write(s)


# ---------- 1. cpu_ops.cpp ----------
p = os.path.join(T, 'csrc', 'cpu_ops.cpp')
src = read(p)
if MARK in src:
    print('1. cpu_ops.cpp     已接入，跳过')
else:
    block = read(BLOCK)
    # 去掉 block 自己的 BOM/换行，直接追加
    src = src.rstrip() + '\r\n\r\n' + block.strip() + '\r\n'
    write(p, src)
    print('1. cpu_ops.cpp     +%d 字符' % len(block))

# ---------- 2. cpu_ops.h ----------
DECL = '''
// ----------------------------------------------------------------------------
// Fused elementwise kernels (2026-09-18).  Training-step profiling on
// R5-4500U (30.49M AR model, batch=8 T=256) showed GEMM already at 92% of a
// same-shape pure-GEMM baseline while elementwise ops burn 20.8% of the step
// for <5% of the FLOPs -- the eager RMSNorm expands to six aten ops per call.
// These kernels fuse those chains into single passes.
// ----------------------------------------------------------------------------
#if defined(__cplusplus)
extern "C" {
#endif
long long fused_rmsnorm_fwd_cpu(
    const float* x, const float* res, const float* weight, float* out, float* xs,
    long long M, long long D, float eps
);
long long fused_rmsnorm_bwd_cpu(
    const float* x, const float* weight, const float* dout, const float* xs,
    float* dx, float* dw, long long M, long long D
);
long long fused_swiglu_fwd_cpu(const float* gate, const float* up, float* out, long long n);
long long fused_swiglu_bwd_cpu(
    const float* gate, const float* up, const float* dout, float* dgate, float* dup, long long n
);
long long fused_add_scale_cpu(
    const float* x, const float* y, const float* scale, float* out, long long M, long long D
);
#if defined(__cplusplus)
}
#endif
'''
p = os.path.join(T, 'csrc', 'cpu_ops.h')
src = read(p)
if 'fused_rmsnorm_fwd_cpu' in src:
    print('2. cpu_ops.h       已接入，跳过')
else:
    # 插到最后一个 #endif 之前
    idx = src.rstrip().rfind('#endif')
    src = src[:idx] + DECL + '\r\n' + src[idx:]
    write(p, src)
    print('2. cpu_ops.h        +%d 字符' % len(DECL))

# ---------- 3. pythonInterface.cpp ----------
BIND = '''
// ---- fused elementwise kernels (2026-09-18) ----
long long cfused_rmsnorm_fwd_cpu(
    const float* x, const float* res, const float* weight, float* out, float* xs,
    long long M, long long D, float eps
) {
    return fused_rmsnorm_fwd_cpu(x, res, weight, out, xs, M, D, eps);
}

long long cfused_rmsnorm_bwd_cpu(
    const float* x, const float* weight, const float* dout, const float* xs,
    float* dx, float* dw, long long M, long long D
) {
    return fused_rmsnorm_bwd_cpu(x, weight, dout, xs, dx, dw, M, D);
}

long long cfused_swiglu_fwd_cpu(const float* gate, const float* up, float* out, long long n) {
    return fused_swiglu_fwd_cpu(gate, up, out, n);
}

long long cfused_swiglu_bwd_cpu(
    const float* gate, const float* up, const float* dout, float* dgate, float* dup, long long n
) {
    return fused_swiglu_bwd_cpu(gate, up, dout, dgate, dup, n);
}

long long cfused_add_scale_cpu(
    const float* x, const float* y, const float* scale, float* out, long long M, long long D
) {
    return fused_add_scale_cpu(x, y, scale, out, M, D);
}
'''
p = os.path.join(T, 'csrc', 'pythonInterface.cpp')
src = read(p)
if 'cfused_rmsnorm_fwd_cpu' in src:
    print('3. pythonInterface 已接入，跳过')
else:
    # pythonInterface.cpp 里所有 C 入口都在 extern "C" { ... } 块内
    # 找到最后一个单独成行的 "}" （extern "C" 的结束）
    lines = src.split('\n')
    last = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '}':
            last = i
            break
    if last is None:
        raise SystemExit('找不到 extern "C" 的结束位置')
    lines.insert(last, BIND)
    # 修正：把新代码插到 "}" 之前
    src = '\n'.join(lines)
    # 需要包含 cpu_ops.h —— 确认已有
    if '#include "cpu_ops.h"' not in src:
        src = '#include "cpu_ops.h"\n' + src
        print('   （补上了 #include "cpu_ops.h"）')
    write(p, src)
    print('3. pythonInterface +%d 字符' % len(BIND))

# ---------- 4. export.def ----------
EXPORTS = '''    cfused_rmsnorm_fwd_cpu
    cfused_rmsnorm_bwd_cpu
    cfused_swiglu_fwd_cpu
    cfused_swiglu_bwd_cpu
    cfused_add_scale_cpu
'''
p = os.path.join(T, 'build_manual', 'export.def')
src = read(p)
if 'cfused_rmsnorm_fwd_cpu' in src:
    print('4. export.def      已接入，跳过')
else:
    src = src.rstrip() + '\n' + EXPORTS
    write(p, src)
    print('4. export.def      +%d 字符' % len(EXPORTS))

print()
print('接入完成。')
