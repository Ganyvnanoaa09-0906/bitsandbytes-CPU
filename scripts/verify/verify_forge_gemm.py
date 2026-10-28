# -*- coding: utf-8 -*-
"""verify_forge_gemm.py —— DLL 的四种布局 vs torch，在【真实预训练形状】上对齐。

形状来自 debug_mm_shapes.py 抓到的实况（v4_bos/ckpt_13000, B=8 T=256 L=10 d=512）：
  w12 前向  2048 x 2736 x 512
  w3  前向  2048 x 512 x 1368
  w12 gradW 2736 x 2048 x 512
  w3  gradW 512 x 2048 x 1368
  方阵族     2048 x 512 x 512
  head      2048 x 512 x 1024 / 128 x 512 x 1024

每种布局都比 max_abs / 相对峰值，阈值 1e-4（fp32 累加顺序不同带来的正常偏差）。
★ 顺序也是错的来源：这个内核沿 R 分块累加，跟 torch 的累加顺序【不同】，
  所以不可能 bit 级一致。判据只能是相对误差。
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os
import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib import forge_gemm

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)
FG.set_threads(TH)
print('=' * 92)
print('forge_gemm 验证   线程=%d   %s' % (TH, FG.info()))
print('=' * 92)
print()

# (名字, I, J, R) —— 即 C 的形状和归约维
SHAPES = [
    ('w12 前向',   2048, 2736,  512),
    ('w3  前向',   2048,  512, 1368),
    ('w12 gradW',  2736,  512, 2048),
    ('w3  gradW',   512, 1368, 2048),
    ('方阵 o_proj', 2048,  512,  512),
    ('head gen',   2048, 1024,  512),
    ('head think',  128, 1024,  512),
    ('小尾巴',       13,   40,   29),
    ('非整块',      131,   67, 1000),
]


def build(A_shape_kind, I, J, R, layout):
    """按布局造 A/B，返回 (A, B, C_ref)。"""
    if layout == 'NT':
        A = torch.randn(I, R); B = torch.randn(J, R)
    elif layout == 'NN':
        A = torch.randn(I, R); B = torch.randn(R, J)
    elif layout == 'TN':
        A = torch.randn(R, I); B = torch.randn(R, J)
    else:
        A = torch.randn(R, I); B = torch.randn(J, R)
    return A, B


bad = 0
print('  %-14s %-5s %-18s %10s %10s %12s  %s'
      % ('形状', '布局', 'I x J x R', 'max_abs', '峰值', '相对', '结论'))
print('  ' + '-' * 88)
for name, I, J, R in SHAPES:
    for layout in ('NT', 'NN', 'TN', 'TT'):
        A, B = build(None, I, J, R, layout)
        ref = FG.torch_ref(A, B, layout)
        got = FG.mm(A, B, layout)
        maxd = (got - ref).abs().max().item()
        peak = ref.abs().max().item()
        rel = maxd / max(peak, 1e-30)
        ok = rel < 1e-4 and got.shape == ref.shape
        if not ok:
            bad += 1
        print('  %-14s %-5s %-18s %10.3e %10.3e %12.3e  %s'
              % (name[:14], layout, '%d x %d x %d' % (I, J, R), maxd, peak, rel,
                 'OK' if ok else '**失败**'))
    print()

# ---- Linear 封装的三件套：前向 / grad_x / grad_W ----
print('  ---- Linear 三件套（对照 F.linear 的 autograd） ----')
for name, M, N, K in (('w12', 2048, 2736, 512), ('w3', 2048, 512, 1368),
                      ('方阵', 2048, 512, 512)):
    x = torch.randn(M, K, requires_grad=True)
    W = torch.randn(N, K, requires_grad=True)
    y_ref = torch.nn.functional.linear(x, W)
    y_ref.sum().backward()
    gx_ref, gW_ref = x.grad.clone(), W.grad.clone()

    y = FG.linear_fwd(x.detach(), W.detach())
    gx, gW = FG.linear_bwd(torch.ones_like(y), x.detach(), W.detach())
    for tag, got, exp in (('前向', y, y_ref), ('grad_x', gx, gx_ref), ('grad_W', gW, gW_ref)):
        rel = (got - exp).abs().max().item() / max(exp.abs().max().item(), 1e-30)
        ok = rel < 1e-4
        if not ok:
            bad += 1
        print('  %-6s %-8s 相对误差 %.3e  %s' % (name, tag, rel, 'OK' if ok else '**失败**'))

print()
print('==== %s ====' % ('全部通过' if bad == 0 else '**有 %d 项失败**' % bad))
sys.exit(1 if bad else 0)
