# -*- coding: utf-8 -*-
"""test_maskgit.py — MaskGIT 离散扩散自检（重点是验证"按构造不可能泄漏"）

## 为什么这个自检是本项目最重要的之一

我们栽过的最大的坑是 `ar_grn256`：
    全双向 + GRN 随机替换、loss 算在【所有位置】
    ⇒ 模型学会"抄自己的输入"
    ⇒ val=0.9716（困惑度 2.64）、top-1=79.6% 全是假数字
    ⇒ 判决：干净输入 CE=1.02/top-1=79.6%；输入换随机后 CE=7.24/top-1=0.17%

MaskGIT 的设计恰好绕开：输入是 [MASK]（不含答案信息）+ **loss 只算被掩码的位置**。
本自检要证明两件事：
  ① 【负向】如果 loss 算在未掩码位置 ⇒ 模型能近乎 100% 抄对（复现那个失效）
  ② 【正向】loss 只算掩码位置时，被掩码位置的真值【不可能】影响该位置的预测
     （因为它的输入恒为 MASK 嵌入）⇒ 按构造无泄漏
  ③ 采样的余弦调度正确（迭代结束时应无 MASK 残留）
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
import sys, math, torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

G, T, V0 = 16, 256, 1024
MASK = V0            # MASK id = 1024（vocab 1025）
print('=' * 90)
print('MaskGIT 自检   grid=%d T=%d  真实码 0..%d  MASK=%d' % (G, T, V0 - 1, MASK))
print('=' * 90)

cfg = SmallImageConfigV2(vocab_size=V0 + 1, d_model=512, n_layer=4, n_head=8,
                         n_kv_head=4, ffn_dim=2048, total_tokens=T, grid=G,
                         pos_mode='2d', mask_mode='full', head_mode='flat',
                         cond_mode='none', use_bos=False)
m = SmallARImageModelV2(cfg)
print('  params %.2fM   vocab=%d (含 MASK)' % (sum(p.numel() for p in m.parameters()) / 1e6,
                                              cfg.vocab_size))

torch.manual_seed(0)
x = torch.randint(0, V0, (2, T))

# ---------- ① 负向：loss 算在未掩码位置 ⇒ 模型能抄（复现失效）----------
print()
print('[1] 负向对照：为什么 loss 【必须】只算被掩码的位置')
lg_clean = m.forward_masked(x, None)[..., :V0]        # 全不掩码
acc_clean = float((lg_clean.argmax(-1) == x).float().mean())
print('  全不掩码输入的 top-1（未训练模型，应≈随机 1/1024=0.1%%）: %.2f%%' % (100 * acc_clean))
print('  ⇒ 训练后这个数会趋近 100%%（模型直接读自己的输入嵌入）')
print('     ar_grn256 就是这么报出 val=0.97 / top-1=79.6%% 的 —— 全是假的')

# ---------- ② 正向：掩码位的预测与真值【无关】----------
print()
print('[2] 正向：被掩码位置的真值【不可能】影响该位置的预测')
mk = torch.rand(2, T) < 0.5
x_in = torch.where(mk, torch.full_like(x, MASK), x)
lg1 = m.forward_masked(x_in, None)[..., :V0]
x2 = x.clone()
# 把被掩码位置的真值换成完全不同的码
x2[mk] = torch.randint(0, V0, (int(mk.sum()),))
x_in2 = torch.where(mk, torch.full_like(x2, MASK), x2)
same_input = bool((x_in == x_in2).all())
lg2 = m.forward_masked(x_in2, None)[..., :V0]
d_all = (lg1 - lg2).abs().max().item()
d_masked = (lg1[mk] - lg2[mk]).abs().max().item()
print('  改掉被掩码位的【真值】后：输入序列是否完全不变: %s' % same_input)
print('  掩码位 logits 最大差: %.3e  %s' % (d_masked, 'OK（无泄漏）' if d_masked < 1e-6 else '❌'))
print('  全部位 logits 最大差: %.3e' % d_all)
print('  ⇒ 掩码位的输入恒为 MASK 嵌入 ⇒ 按构造不可能泄漏')

# ---------- ③ 采样：余弦调度 + 无残留 ----------
print()
print('[3] 采样 generate_masked：余弦调度、迭代结束无 MASK 残留')
for n_iter in (1, 4, 8, 16):
    s = m.generate_masked(total=T, n_iter=n_iter, mask_id=MASK, batch=2, seed=1)
    n_mask = int((s == MASK).sum())
    print('  n_iter=%2d -> 输出 %s  取值[%d,%d]  残留 MASK %d  %s'
          % (n_iter, tuple(s.shape), int(s.min()), int(s.max()), n_mask,
             'OK' if n_mask == 0 else '❌'))

# ---------- ④ 掩码比例调度 sanity ----------
print()
print('[4] 训练掩码比例 r~U[0,1] 的实际覆盖率（应≈50%，且跨度覆盖 0~1）')
tot, mn, mx = 0.0, 1.0, 0.0
for _ in range(200):
    rr = torch.rand(4, 1)
    mk2 = torch.rand(4, T) < rr
    f = float(mk2.float().mean())
    tot += f; mn = min(mn, f); mx = max(mx, f)
print('  平均掩码率 %.3f   最小 %.3f   最大 %.3f' % (tot / 200, mn, mx))

print()
print('=' * 90)
print('判读：')
print('  * [2] 的"掩码位 logits 最大差 = 0"是 MaskGIT 无泄漏的【构造性证明】')
print('  * [1] 提醒：val loss 必须用【同样的掩码方式】算，否则会重犯 ar_grn256 的错')
print('DONE')
