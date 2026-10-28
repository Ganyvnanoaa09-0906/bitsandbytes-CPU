# -*- coding: utf-8 -*-
"""test_fused_kernels.py — 融合核的正确性 + 性能验证

正确性：与 PyTorch eager 参考实现对比（前向 + 反向梯度）
性能：与 eager 参考对比单次调用的耗时
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 80)
print('bnb 融合 elementwise 核 —— 正确性 + 性能')
print('=' * 80)

from scripts.lib import fused_cpu
print()
print('DLL 可用:', fused_cpu.available())
if not fused_cpu.available():
    sys.exit('需要先编译 bnb（含融合核）')

B, T, D = 8, 256, 512
M = B * T
HID = 1368   # SwiGLU hidden（ffn_dim 2048 时 = 2048*2/3 向上取 8 的倍数）


def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


# =================== 1. RMSNorm ===================
print()
print('[1] RMSNorm  —— out = (x+res)/sqrt(mean((x+res)^2)+eps) * w')
x = torch.randn(M, D)
res = torch.randn(M, D)
w = torch.randn(D)
eps = 1e-6


def ref_rms(x, res, w, eps):
    h = x + res
    return w * h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)


ref = ref_rms(x, res, w, eps)
got = fused_cpu.fused_rmsnorm(x, w, res, eps)
d = (ref - got).abs().max().item()
rel = d / ref.abs().max().item()
print('    前向  最大绝对差 %.3e  相对 %.3e  %s' % (d, rel, 'OK' if rel < 1e-5 else 'FAIL'))

# 梯度
x1 = x.clone().requires_grad_(True)
r1 = res.clone().requires_grad_(True)
w1 = w.clone().requires_grad_(True)
ref_rms(x1, r1, w1, eps).sum().backward()

x2 = x.clone().requires_grad_(True)
r2 = res.clone().requires_grad_(True)
w2 = w.clone().requires_grad_(True)
fused_cpu.fused_rmsnorm(x2, w2, r2, eps).sum().backward()

for name, a, b in (('dx', x1.grad, x2.grad), ('dres', r1.grad, r2.grad), ('dw', w1.grad, w2.grad)):
    if a is None or b is None:
        print('    %-6s  缺梯度 (ref=%s fused=%s)' % (name, a is not None, b is not None))
        continue
    dd = (a - b).abs().max().item()
    sc = max(a.abs().max().item(), 1e-9)
    print('    %-6s 最大绝对差 %.3e  相对 %.3e  %s'
          % (name, dd, dd / sc, 'OK' if dd / sc < 1e-4 else 'FAIL'))

# 性能
with torch.no_grad():
    t_ref = bench(lambda: ref_rms(x, res, w, eps))
    t_fus = bench(lambda: fused_cpu.fused_rmsnorm(x, w, res, eps))
print('    eager %.3f ms  vs  融合 %.3f ms  ⇒ 【%.2fx】' % (t_ref * 1e3, t_fus * 1e3, t_ref / t_fus))

# =================== 2. SwiGLU ===================
print()
print('[2] SwiGLU  —— out = silu(gate) * up')
gate = torch.randn(M, HID)
up = torch.randn(M, HID)

ref = F.silu(gate) * up
got = fused_cpu.fused_swiglu(gate, up)
d = (ref - got).abs().max().item()
rel = d / ref.abs().max().item()
print('    前向  最大绝对差 %.3e  相对 %.3e  %s' % (d, rel, 'OK' if rel < 1e-5 else 'FAIL'))

g1 = gate.clone().requires_grad_(True)
u1 = up.clone().requires_grad_(True)
(F.silu(g1) * u1).sum().backward()
g2 = gate.clone().requires_grad_(True)
u2 = up.clone().requires_grad_(True)
fused_cpu.fused_swiglu(g2, u2).sum().backward()
for name, a, b in (('dgate', g1.grad, g2.grad), ('dup', u1.grad, u2.grad)):
    dd = (a - b).abs().max().item()
    sc = max(a.abs().max().item(), 1e-9)
    print('    %-6s 最大绝对差 %.3e  相对 %.3e  %s'
          % (name, dd, dd / sc, 'OK' if dd / sc < 1e-4 else 'FAIL'))

with torch.no_grad():
    t_ref = bench(lambda: F.silu(gate) * up)
    t_fus = bench(lambda: fused_cpu.fused_swiglu(gate, up))
print('    eager %.3f ms  vs  融合 %.3f ms  ⇒ 【%.2fx】' % (t_ref * 1e3, t_fus * 1e3, t_ref / t_fus))

# =================== 3. add_scale ===================
print()
print('[3] add_scale  —— out = x + scale * y   (LayerScale + 残差)')
xx = torch.randn(M, D)
yy = torch.randn(M, D)
ls = torch.full((D,), 1e-2)

ref = xx + ls * yy
got = fused_cpu.fused_add_scale(xx, yy, ls)
d = (ref - got).abs().max().item()
print('    前向  最大绝对差 %.3e  %s' % (d, 'OK' if d < 1e-6 else 'FAIL'))

with torch.no_grad():
    t_ref = bench(lambda: xx + ls * yy)
    t_fus = bench(lambda: fused_cpu.fused_add_scale(xx, yy, ls))
print('    eager %.3f ms  vs  融合 %.3f ms  ⇒ 【%.2fx】' % (t_ref * 1e3, t_fus * 1e3, t_ref / t_fus))

print()
print('=' * 80)
print('DONE')
