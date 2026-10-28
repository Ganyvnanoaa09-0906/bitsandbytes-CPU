# -*- coding: utf-8 -*-
"""test_mhc.py — 验证 mHC 实现

检查四件事：
  ① 双随机性：Sinkhorn 输出必须行和=列和=1，且非负
  ② 恒等映射：初始化时 H_res 应接近单位阵（保持残差性质）
  ③ 非扩张性：谱范数 ≤ 1（这是 mHC 修复 HC 的核心）
  ④ 成本：实测 vs 估算
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
import sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.mhc import sinkhorn, MHCResidual, mhc_cost_estimate

torch.set_num_threads(6)
torch.manual_seed(0)
print('=' * 80)
print('mHC 实现验证')
print('=' * 80)

# ---- ① 双随机性 ----
print()
print('【①】Sinkhorn 输出的双随机性')
x = torch.randn(256, 4, 4) * 2.0
H = sinkhorn(x, 20)
rs = H.sum(-1)
cs = H.sum(-2)
print('  行和: min=%.6f max=%.6f  (应全为 1)' % (rs.min(), rs.max()))
print('  列和: min=%.6f max=%.6f  (应全为 1)' % (cs.min(), cs.max()))
print('  最小元素: %.6f  (应 ≥ 0)' % H.min())
ok1 = (rs - 1).abs().max() < 3e-2 and (cs - 1).abs().max() < 3e-2 and H.min() >= -1e-9
print('  %s' % ('✅ 双随机' if ok1 else '❌ 不满足'))

# ---- ② 恒等映射（初始化时）----
print()
print('【②】初始化时 H_res 应接近单位阵')
m = MHCResidual(n_streams=4, d_model=512, sinkhorn_iters=50)
h = m.init(torch.randn(2, 8, 512))
Hm = m._mix_matrix(h)
print('  对角均值: %.4f   非对角均值: %.4f' % (
    Hm.diagonal(dim1=-2, dim2=-1).mean(), 
    (Hm.sum(-1).sum(-1) - Hm.diagonal(dim1=-2, dim2=-1).sum(-1)).mean() / 3))
print('  与单位阵的最大偏差: %.4f' % (Hm - torch.eye(4)).abs().max())
ok2 = (Hm - torch.eye(4)).abs().max() < 0.3
print('  %s' % ('✅ 接近恒等' if ok2 else '⚠️ 偏离较大（可能影响初期稳定性）'))

# ---- ③ 非扩张性 ----
print()
print('【③】谱范数 ≤ 1（mHC 修复 HC 的核心）')
mags = []
for _ in range(64):
    xx = torch.randn(1, 4, 4) * 3
    HH = sinkhorn(xx, 20)
    mags.append(torch.linalg.matrix_norm(HH, ord=2).max().item())
print('  谱范数: max=%.4f  均值=%.4f' % (max(mags), sum(mags) / len(mags)))
ok3 = max(mags) <= 1.0 + 2e-2
print('  %s' % ('✅ 非扩张（≤1）' if ok3 else '❌ 超过 1'))

# 对照：无约束的自由矩阵
free = torch.randn(64, 4, 4) * 2
print('  对照（自由矩阵）: 谱范数均值 %.4f  max %.4f' % (
    torch.linalg.matrix_norm(free, ord=2, dim=(-2, -1)).mean(),
    torch.linalg.matrix_norm(free, ord=2, dim=(-2, -1)).max()))

# ---- ④ 成本实测 ----
print()
print('【④】成本实测（batch=8, T=256, n=4, d=512）')
h = m.init(torch.randn(8, 256, 512))
out = torch.randn(8, 256, 512)


def one():
    return m.mix(h, out)


for _ in range(3):
    one()
t0 = time.perf_counter()
for _ in range(5):
    one()
t = (time.perf_counter() - t0) / 5
print('  mHC 一次 mix: %.2f ms' % (t * 1e3))
sink_ms, mix_ms, pct = mhc_cost_estimate()
print('  估算 (Sinkhorn %.2f ms + 混合 %.2f ms) = %.2f ms  ⇒ 训练步 %.2f%%'
      % (sink_ms, mix_ms, sink_ms + mix_ms, pct))

# ---- ⑤ 形状与梯度 ----
print()
print('【⑤】形状与前向链路')
x0 = torch.randn(2, 16, 512)
h = m.init(x0)
print('  init:  %s -> %s' % (tuple(x0.shape), tuple(h.shape)))
for i in range(3):
    h = m.mix(h, torch.randn(2, 16, 512))
print('  3 次 mix 后: %s' % (tuple(h.shape),))
print('  read: %s' % (tuple(m.read(h).shape),))
loss = m.read(h).pow(2).mean()
loss.backward()
ng = sum(1 for p in m.parameters() if p.grad is not None)
print('  梯度可达参数: %d/%d  %s' % (ng, sum(1 for _ in m.parameters()),
                                    '✅' if ng == sum(1 for _ in m.parameters()) else '❌'))
print('  mHC 参数量: %d (%.1f k)' % (
    sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters()) / 1e3))

print()
print('=' * 80)
allok = ok1 and ok3
print('结论: %s' % ('✅ 核心性质全部满足，可用' if allok else '❌ 有问题'))
print('DONE')
