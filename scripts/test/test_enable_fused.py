# -*- coding: utf-8 -*-
"""test_enable_fused.py — 验证 enable_fused 模块能用（正确性 + 生效）"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 74)
print('enable_fused 验证')
print('=' * 74)

from scripts.lib import fused_cpu
from scripts.lib import enable_fused
from scripts.lib import small_image_model_v2 as M

B, T = 4, 256
cfg = M.SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=T, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
torch.manual_seed(0)
m = M.SmallARImageModelV2(cfg)
m.eval()
tok = torch.randint(0, 1024, (B, T))

with torch.no_grad():
    _, g0 = m.forward_logits(tok)
g0 = g0[0] if isinstance(g0, (tuple, list)) else g0
sc = g0.abs().max().item()

print()
print('[1] enable() 默认（只 RMSNorm）')
enable_fused.enable()
print('    status:', enable_fused.status())
with torch.no_grad():
    _, g1 = m.forward_logits(tok)
g1 = g1[0] if isinstance(g1, (tuple, list)) else g1
print('    logits 相对差 %.3e  %s' % ((g1 - g0).abs().max().item() / sc,
                                     'OK' if (g1 - g0).abs().max().item() / sc < 1e-4 else 'FAIL'))

print()
print('[2] 全开')
enable_fused.enable(swiglu=True, block=True)
print('    status:', enable_fused.status())
with torch.no_grad():
    _, g2 = m.forward_logits(tok)
g2 = g2[0] if isinstance(g2, (tuple, list)) else g2
print('    logits 相对差 %.3e  %s' % ((g2 - g0).abs().max().item() / sc,
                                     'OK' if (g2 - g0).abs().max().item() / sc < 1e-4 else 'FAIL'))

print()
print('[3] 反向也能跑')
m.train()
_, g3 = m.forward_logits(tok)
g3 = g3[0] if isinstance(g3, (tuple, list)) else g3
loss = F.cross_entropy(g3.reshape(-1, 1024), tok.reshape(-1))
loss.backward()
ng = sum(1 for p in m.parameters() if p.grad is not None)
na = sum(1 for _ in m.parameters())
print('    loss=%.4f  梯度 %d/%d  %s' % (float(loss), ng, na, 'OK' if ng == na else 'FAIL'))

print()
print('[4] disable()')
enable_fused.disable()
print('    status:', enable_fused.status())
m.eval()
with torch.no_grad():
    _, g4 = m.forward_logits(tok)
g4 = g4[0] if isinstance(g4, (tuple, list)) else g4
d = (g4 - g0).abs().max().item()
print('    关闭后与基线差 %.3e  %s' % (d, 'OK（完全恢复）' if d == 0 else 'WARN'))

print()
print('=' * 74)
print('DONE')
