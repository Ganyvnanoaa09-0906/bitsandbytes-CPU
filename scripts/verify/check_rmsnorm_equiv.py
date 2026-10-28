# -*- coding: utf-8 -*-
"""check_rmsnorm_equiv.py —— 手写 RMSNorm vs F.rms_norm 的数值等价性验证。

背景：small_image_model_v2.RMSNorm.forward 是
    self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
一句话展开成 6 个 kernel（pow/mean/add/rsqrt/mul/mul），每个都把张量整个读写一遍。
算子剖析显示这类基础算子合计占了整步的 26%，而 aten::mm 只有 47%。

改成 F.rms_norm（torch>=2.4 提供）是一个 kernel 搞定。

★ 动它之前必须先证明【数值一致】——训练跑了 13000 步，任何漂移都会毁掉已有进度。
  这里不只比 max_abs，还要按真实量级算相对误差，并检查 fp32 下的退化输入
  （全零、极大值、极小值、含 inf/nan 的边界）。
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)
print('torch %s' % torch.__version__)
print()

if not hasattr(F, 'rms_norm'):
    print('!! 这个 torch 没有 F.rms_norm，方案不可行')
    sys.exit(2)
print('F.rms_norm 存在 ✓')
print()


def old_rmsnorm(x, w, eps=1e-6):
    return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def new_rmsnorm(x, w, eps=1e-6):
    return F.rms_norm(x, (x.shape[-1],), w, eps)


# ---- 真实形状：B=8, T=256, d=512；以及 q/k norm 用的 (B,H,T,hd) ----
CASES = [
    ('ln1/ln2  (8,256,512)', (8, 256, 512), 512),
    ('q_norm   (8,8,256,64)', (8, 8, 256, 64), 64),
    ('k_norm   (8,2,256,64)', (8, 2, 256, 64), 64),
    ('小 d=16', (4, 7, 16), 16),
]

allok = True
print('  %-24s %-10s %-12s %-12s %-12s %s' % ('形状', '量级', 'max_abs', '相对峰值', 'cos', '结论'))
print('  ' + '-' * 92)
for name, shp, d in CASES:
    for scale, sname in ((1.0, '正常'), (1e-3, '小'), (1e3, '大')):
        x = torch.randn(*shp) * scale
        w = torch.randn(d) * 0.5 + 1.0
        a = old_rmsnorm(x, w.clone())
        b = new_rmsnorm(x, w.clone())
        maxabs = (a - b).abs().max().item()
        peak = a.abs().max().item()
        cos = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
        rel = maxabs / max(peak, 1e-30)
        ok = rel < 1e-5 and cos > 1 - 1e-7
        allok &= ok
        print('  %-24s %-10s %-12.3e %-12.3e %-12.9f %s'
              % (name, sname, maxabs, rel, cos, 'OK' if ok else '**不一致**'))
    print()

# ---- 边界：全零 / 常数 / 含 inf-1e30 的大值 ----
print('  ---- 退化输入 ----')
EDGE = [
    ('全零', lambda shp, d: torch.zeros(*shp)),
    ('全常数 1', lambda shp, d: torch.ones(*shp)),
    ('全常数 1e-20', lambda shp, d: torch.full(shp, 1e-20)),
    ('单个大值', lambda shp, d: torch.zeros(*shp).index_put_(
        (torch.zeros(1, dtype=torch.long),), torch.tensor([1e30]))),
]
for name, mk in EDGE:
    shp, d = (4, 8, 16), 16
    x = mk(shp, d)
    w = torch.ones(d)
    a = old_rmsnorm(x, w.clone())
    b = new_rmsnorm(x, w.clone())
    both_nan = bool(torch.isnan(a).any()) and bool(torch.isnan(b).any())
    if both_nan:
        print('  %-20s 两边都出 NaN -> 行为一致' % name)
        continue
    maxabs = (a - b).abs().max().item()
    ok = maxabs < 1e-5
    allok &= ok
    print('  %-20s max_abs=%.3e  %s' % (name, maxabs, 'OK' if ok else '**不一致**'))

# ---- 反向梯度也要一致（训练是靠反向的） ----
print()
print('  ---- 反向梯度 ----')
x1 = torch.randn(8, 256, 512, requires_grad=True)
x2 = x1.detach().clone().requires_grad_(True)
w1 = (torch.randn(512) * 0.5 + 1).requires_grad_(True)
w2 = w1.detach().clone().requires_grad_(True)
old_rmsnorm(x1, w1).pow(2).sum().backward()
new_rmsnorm(x2, w2).pow(2).sum().backward()
gx = (x1.grad - x2.grad).abs().max().item() / max(x1.grad.abs().max().item(), 1e-30)
gw = (w1.grad - w2.grad).abs().max().item() / max(w1.grad.abs().max().item(), 1e-30)
ok = gx < 1e-5 and gw < 1e-5
allok &= ok
print('    grad_x 相对误差 %.3e   grad_w 相对误差 %.3e   %s'
      % (gx, gw, 'OK' if ok else '**不一致**'))

print()
print('==== %s ====' % ('数值等价，可以替换' if allok else '**有差异，不能直接替换**'))
sys.exit(0 if allok else 1)
