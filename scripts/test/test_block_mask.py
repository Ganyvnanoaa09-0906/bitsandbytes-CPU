# -*- coding: utf-8 -*-
"""test_block_mask.py — 验证 VAR 风格 block-causal mask 的正确性

要验证：
  ① bound(i) 的计算正确（最小的 4 的幂 ≥ i+1）
  ② mask 的形状和语义正确（块内双向、块间因果）
  ③ Z-order 排列下，"4^k 前缀" 是否真的是 2^k×2^k 的方形子块
     （这是"块 = 尺度"的前提，必须成立）
  ④ 模型能跑起来，且 block mask 与 causal mask 数值上不同
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
torch.set_num_threads(6)
print('=' * 80)
print('VAR 风格 block-causal mask 验证')
print('=' * 80)

from scripts.lib import small_image_model_v2

T = 256
side = 16

# ---------- ① bound ----------
cfg = M.SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=2, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=T, grid=side, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='zorder', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True,
    mask_mode='block')
attn = M.SelfAttention(cfg)

m = attn._block_mask(T, T, torch.device('cpu'))[0, 0]     # (T, T)
print()
print('[1] bound(i) 与 mask 语义')
print('    期望: bound = 最小的 4 的幂 ≥ i+1')
print('    %-8s %-10s %-10s %-12s' % ('i', 'i+1', 'bound', '可见前缀长'))
print('    ' + '-' * 44)
for i in [0, 1, 3, 4, 5, 15, 16, 20, 63, 64, 100, 255]:
    vis = int(m[i].sum().item())
    # 反推 bound
    b = 1
    while b < i + 1:
        b *= 4
    print('    %-8d %-10d %-10d %-12d %s' % (i, i + 1, b, vis, 'OK' if vis == b else 'FAIL'))

# 块内双向检查
print()
print('[2] 块内双向 / 块间因果')
ok_sym = ok_causal = 0
for b in (1, 4, 16, 64, 256):
    lo, hi = b // 4, b          # 上一块的末尾 .. 本块的末尾
    if lo < 0:
        lo = 0
    for i in range(lo, hi):
        for j in range(lo, min(hi, T)):
            if m[i, j] == m[j, i]:
                ok_sym += 1
    # 块内 i 应能看到同块的所有 j
print('    同一块内 mask 对称（双向）:', ok_sym)
print('    交叉检查：i=100, j=20 可见吗:', bool(m[100, 20]))   # 100 在块 256，20 在块 64 ⇒ 可见
print('              i=20, j=100 可见吗:', bool(m[20, 100]))   # 20 在块 64，100 在块 256 ⇒ 不可见
print('              i=5, j=6  可见吗:', bool(m[5, 6]))
print('              i=6, j=5  可见吗:', bool(m[6, 5]))

# ---------- ③ Z-order 前缀是否为方形子块 ----------
print()
print('[3] Z-order 排列下，4^k 前缀是否为 2^k×2^k 的方形子块')
order = M.make_token_order(side, 'zorder')
print('    zorder 排列前 8 个:', order[:8])
for k in (1, 2, 3):
    n = 4 ** k          # 1, 4, 16, 64
    idxs = order[:n]
    rows = sorted(set(i // side for i in idxs))
    cols = sorted(set(i % side for i in idxs))
    is_square = (len(rows) == len(cols)) and (rows == list(range(rows[0], rows[-1] + 1)))
    print('    前 %3d 个token: 行范围 [%d,%d] (%d 行)  列范围 [%d,%d] (%d 列)  %s'
          % (n, rows[0], rows[-1], len(rows), cols[0], cols[-1], len(cols),
             '方形 ✅' if is_square and len(rows) == 2 ** k else '⚠️ 非方形'))
print('    （若成立 ⇒ "块 = 尺度" 的前提满足）')

# ---------- ④ 模型前向 ----------
print()
print('[4] 模型前向：block mask vs causal mask')
tok = torch.randint(0, 1024, (2, T))
res = {}
for mode in ('causal', 'block'):
    cfg2 = M.SmallImageConfigV2(
        vocab_size=1024, d_model=512, n_layer=2, n_head=8, n_kv_head=4,
        ffn_dim=2048, total_tokens=T, grid=side, think_tokens=16,
        text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
        pos_mode='2d', token_order='zorder', head_mode='flat', n_cluster=32,
        hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True,
        mask_mode=mode)
    torch.manual_seed(0)
    mm = M.SmallARImageModelV2(cfg2)
    mm.eval()
    with torch.no_grad():
        _, g = mm.forward_logits(tok)
    g = g[0] if isinstance(g, (tuple, list)) else g
    res[mode] = g
    print('    %-8s 前向 OK   logits %s' % (mode, tuple(g.shape)))
d = (res['causal'] - res['block']).abs().max().item()
print('    两者最大绝对差 %.4e  ⇒ %s' % (d, 'block mask 生效' if d > 1e-3 else '⚠️ 未生效'))

# ---------- ⑤ 序列长度与步数 ----------
print()
print('[5] 复杂度对比')
print('    光栅序 causal : 自回归步数 %d, attention 掩码下三角' % T)
print('    Z-order block : 自回归步数 %d (块 = 1,4,16,64,256)'
      % sum(1 for b in (1, 4, 16, 64, 256) if b <= T))
print('    ⇒ 推理步数减少 %.0f×' % (T / 5))

print()
print('=' * 80)
print('DONE')
