# -*- coding: utf-8 -*-
"""debug_mhc.py — 查清 mHC 的两个问题
  ① 有限迭代 Sinkhorn 的收敛性
  ② 初始化为什么变成"反恒等"
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys
import torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from mhc import sinkhorn, MHCResidual

torch.set_num_threads(4)
torch.manual_seed(0)
print('=' * 76)
print('调试 mHC')
print('=' * 76)

# ---- ① 收敛性 ----
print()
print('【①】Sinkhorn 收敛性（logits = randn*2, 4x4, 256 个）')
x = torch.randn(256, 4, 4) * 2.0
for it in (5, 10, 20, 30, 50, 100, 200):
    H = x.exp()
    for _ in range(it):
        H = H / H.sum(-1, keepdim=True).clamp_min(1e-9)
        H = H / H.sum(-2, keepdim=True).clamp_min(1e-9)
    rs = (H.sum(-1) - 1).abs().max().item()
    cs = (H.sum(-2) - 1).abs().max().item()
    print('  iters=%-4d  行和误差 %.2e  列和误差 %.2e' % (it, rs, cs))
print('  ⇒ 交替归一化收敛慢；必须迭代到列和也满足')

# 试"先归一化再迭代"的两阶段
print()
print('  改进：先各归一化一次，再迭代（每轮末尾都做行+列）')
for it in (5, 10, 20, 30, 50):
    H = x.exp()
    H = H / H.sum(-1, keepdim=True).clamp_min(1e-9)
    H = H / H.sum(-2, keepdim=True).clamp_min(1e-9)
    for _ in range(it):
        H = H / H.sum(-2, keepdim=True).clamp_min(1e-9)
        H = H / H.sum(-1, keepdim=True).clamp_min(1e-9)
    rs = (H.sum(-1) - 1).abs().max().item()
    cs = (H.sum(-2) - 1).abs().max().item()
    print('  iters=%-4d  行和误差 %.2e  列和误差 %.2e' % (it, rs, cs))

# ---- ② 初始化之谜 ----
print()
print('【②】初始化之谜')
m = MHCResidual(n_streams=4, d_model=512)
print('  eye_logit_vec =', m.eye_logit_vec.detach().cpu().tolist())
print('  eye_logit_vec 形状:', tuple(m.eye_logit_vec.shape))
print('  alpha =', float(m.alpha))
print('  phi.weight 绝对值和:', float(m.phi.weight.abs().sum()))

h = m.init(torch.randn(2, 8, 512))
B, T, n, d = h.shape
xf = m.norm(h.reshape(B, T, n * d))
dyn = m.phi(xf) * m.alpha
print('  dyn 绝对值和: %.6f  (应为 0)' % float(dyn.abs().sum()))
logits = dyn + m.eye_logit_vec
print('  logits[0,0] =', logits[0, 0].detach().cpu().tolist())

# 分步看 sinkhorn
H = logits[0, 0].exp()
print('  exp(logits)[0,0] 行和:', H.sum(-1).detach().cpu().tolist())
H1 = H / H.sum(-1, keepdim=True).clamp_min(1e-9)
print('  行归一化后 行和:', H1.sum(-1).detach().cpu().tolist())
H2 = H1 / H1.sum(-2, keepdim=True).clamp_min(1e-9)
print('  再列归一化后 列和:', H2.sum(-2).detach().cpu().tolist())
H3 = sinkhorn(logits, 20)[0, 0]
print('  Sinkhorn(20) 结果[0,0]:')
for r in range(4):
    print('    ', ['%.4f' % v for v in H3[r].detach().cpu().tolist()])
print('  对角:', ['%.4f' % v for v in H3.diagonal().detach().cpu().tolist()])

# ---- ③ 直接测试 mix 的形状 ----
print()
print('【③】mix 的形状')
Hm = m._mix_matrix(h)
print('  _mix_matrix 返回形状:', tuple(Hm.shape), '(应为 B,T,n,n)')
try:
    h2 = torch.bmm(Hm, h.reshape(2 * 8, 4, 512))
    print('  bmm 需要 3D，但 Hm 是 4D => 必须 reshape')
except Exception as e:
    print('  bmm 直接调用失败:', str(e)[:60])
h2 = torch.bmm(Hm.reshape(2 * 8, 4, 4), h.reshape(2 * 8, 4, 512))
print('  reshape 后 bmm 成功:', tuple(h2.shape))
print('DONE')
