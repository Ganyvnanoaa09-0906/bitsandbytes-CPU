# -*- coding: utf-8 -*-
"""test_ms_mask.py — 多尺度 block mask 与棋盘格 mask 的自检"""
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
import sys, torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

PATCH = (1, 2, 4, 8, 16)
T = sum(p * p for p in PATCH)
print('=' * 78)
print('多尺度 mask 自检   patch_nums=%s  总长 T=%d' % (PATCH, T))
print('=' * 78)

cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8,
                         n_kv_head=4, ffn_dim=2048, total_tokens=T, grid=16,
                         mask_mode='msblock', patch_nums=PATCH, pos_mode='ms')
m = SmallARImageModelV2(cfg)
print('  params %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6))

att = m.blocks[0].attn
s = torch.tensor([i for i, pn in enumerate(PATCH) for _ in range(pn * pn)])

print()
print('[1] msblock（尺度内双向 + 跨尺度因果）')
mask = att._ms_block_mask(T, torch.device('cpu'))
print('  形状', tuple(mask.shape), 'dtype', mask.dtype)
begins = []
cur = 0
for pn in PATCH:
    begins.append((cur, cur + pn * pn)); cur += pn * pn
print('  尺度边界', begins)
for i, (a, b) in enumerate(begins):
    if b - a <= 1:
        print('  尺度 %d (%2d tok): 单 token，跳过' % (i, PATCH[i]))
        continue
    print('  尺度 %d (pn=%2d, %3d tok): 尺度内全可见 = %s'
          % (i, PATCH[i], b - a, bool(mask[0, 0, a:b, a:b].all())))
lo, hi = begins[0][0], begins[-1][0]
print('  高尺度(位置%d) 看到低尺度(位置%d): %s' % (hi, lo, bool(mask[0, 0, hi, lo])))
print('  低尺度(位置%d) 看到高尺度(位置%d): %s  (应为 False)'
      % (lo, hi, bool(mask[0, 0, lo, hi])))
print('  下三角全 True:', bool(torch.tril(mask[0, 0]).all()))

print()
print('[2] checkerboard（保底方案：严格无泄漏）')
cb = att._checkerboard_mask(T, torch.device('cpu'))
print('  形状', tuple(cb.shape))
a, b = begins[2]
print('  尺度 2 内可见比例 %.3f (应约 0.5)' % cb[0, 0, a:b, a:b].float().mean().item())
print('  高尺度看到低尺度: %s' % bool(cb[0, 0, hi, lo]))
print('  低尺度看到高尺度: %s  (应为 False)' % bool(cb[0, 0, lo, hi]))
# 对角线必须可见（否则自己都看不到自己）
print('  对角线全可见:', bool(torch.diagonal(cb[0, 0]).all()))
# 同尺度相邻（棋盘格异相）必须不可见
same_scale_adj = 0
for i, (a2, b2) in enumerate(begins):
    pn = PATCH[i]
    for r in range(pn):
        for c in range(pn):
            if c + 1 < pn:
                i1, i2 = a2 + r * pn + c, a2 + r * pn + c + 1
                if cb[0, 0, i2, i1]:
                    same_scale_adj += 1
print('  同尺度相邻可见对数: %d  (应为 0 ⇒ 严格无泄漏)' % same_scale_adj)

print()
print('[3] 真跑一次前向（确认 mask + ms 位置编码被接受）')
for mode in ('msblock', 'causal'):
    cfg2 = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8,
                              n_kv_head=4, ffn_dim=2048, total_tokens=T, grid=16,
                              mask_mode=mode, patch_nums=PATCH, pos_mode='ms')
    m2 = SmallARImageModelV2(cfg2)
    m2.eval()
    x = torch.randint(0, 1024, (2, T))
    out = m2.forward_logits(x)

    def first_tensor(o):
        if torch.is_tensor(o):
            return o
        if isinstance(o, (tuple, list)):
            for e in o:
                t = first_tensor(e)
                if t is not None:
                    return t
        return None

    lg = first_tensor(out)
    assert lg is not None, 'forward_logits 返回值里找不到 tensor: %r' % (type(out),)
    print('  %-8s -> %s  有限值 %s' % (mode, tuple(lg.shape), bool(torch.isfinite(lg).all())))

print()
print('[4] 长度不匹配必须报错（防止静默退化）')
try:
    att._ms_block_mask(256, torch.device('cpu'))
    print('  ❌ 没有报错 —— 会静默用错 mask')
except ValueError as e:
    print('  ✓ 报错:', str(e)[:80])
print()
print('DONE')
