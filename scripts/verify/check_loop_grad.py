# -*- coding: utf-8 -*-
"""check_loop_grad.py — 确认循环模块在 L>1 时能拿到梯度

L=1 时 step_enc / cross_res 都不会被调用（只在 t>0 用），
所以梯度为 0 是正常的。这里验证 L=3 时梯度是否可达。
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
import sys
import torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
from scripts.lib.loop_transformer import LoopConfig

torch.set_num_threads(6)
torch.manual_seed(7)
print('=' * 70)
print('循环模块梯度检查')
print('=' * 70)

cfg = SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=256, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=True, coarse_grid=4, loop_start=4, loop_end=7, loop_times=1,
    fuse_qkv=False, use_bos=True)
tok = torch.randint(0, 1024, (2, 256))


def unwrap(t):
    return t[0] if isinstance(t, (tuple, list)) else t


for L in (1, 3):
    torch.manual_seed(7)
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_L=L, loop_L_max=L, loop_random=False))
    m.train()
    t, g = m.forward_logits(tok)
    g = unwrap(g)
    loss = torch.nn.functional.cross_entropy(g.reshape(-1, 1024), tok.reshape(-1))
    loss.backward()
    print()
    print('L=%d   loss=%.4f' % (L, float(loss)))
    for name, p in m.loop_mods.named_parameters():
        gn = 0.0 if p.grad is None else float(p.grad.abs().mean())
        print('  %-24s shape %-16s |grad|均值 %.3e  %s'
              % (name, tuple(p.shape), gn, 'OK' if gn > 0 else '(未使用)'))

print()
print('判读: L=1 时梯度为 0 是【正常的】（循环模块只在 t>0 被调用）')
print('      L=3 时应该有非零梯度')
print('DONE')
