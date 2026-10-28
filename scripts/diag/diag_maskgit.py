# -*- coding: utf-8 -*-
"""diag_maskgit.py — 诊断 ar_mg256 的 train loss 崩塌 / val 爆炸

现象（云端日志）：
    step 2501   train=5.5859   val=6.9959
    step 3001   train=3.5004   val=7.0390
    step 3501   train=0.1424   val=16.2503   ← 相变
    step 4001   train=0.0375   val=16.1585
train CE 0.0375（困惑度 1.04）而 val CE 16.25（模型"极度自信且极度错误"）。
两者在 step 3000~3500 之间同时相变。

本脚本用拉回来的 ckpt_4000 直接测，待测假设：
  H1 模型在【未掩码位置】上抄自己（ar_grn256 的老毛病）→ 测该位置的 top-1
  H2 任务本身太容易（VQ token 太冗余，50% 可见就几乎确定）→ 用【新随机掩码】在
     验证集上重测 loss，若也很低则 H2 成立
  H3 掩码位置的真值仍在影响预测（结构性泄漏）→ 改真值看 logits 差不差
  H4 val 的固定掩码恰好是病态分布 → 测不同掩码率下的 loss 曲线

★ 最关键的判别是 H1 与 H2：
   若未掩码位置 top-1 ≈ 100%（抄）且掩码位置 loss 也低 → 泄漏
   若未掩码位置 top-1 正常但掩码位置 loss 低 → 任务本身容易，不是泄漏
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
import sys, math, json, torch
import torch.nn.functional
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CK = r'D:\work\cloud_results\mg256_4000.pt'
TOK = r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt'
T = 256
print('=' * 92)
print('MaskGIT 诊断：ckpt_4000')
print('=' * 92)

ck = torch.load(CK, map_location='cpu', weights_only=False)
cfgd = ck['cfg']
keep = {k: v for k, v in cfgd.items() if k in SmallImageConfigV2.__dataclass_fields__}
cfg = SmallImageConfigV2(**keep)
model = SmallARImageModelV2(cfg)
model.load_state_dict(ck['model'])
model.eval()
V0 = cfg.vocab_size - 1
MASK = cfg.vocab_size - 1
print('  step=%s  vocab=%d (MASK=%d)  mask_mode=%s  use_bos=%s'
      % (ck.get('step'), cfg.vocab_size, MASK, cfg.mask_mode, cfg.use_bos))

tok = torch.load(TOK, map_location='cpu')
val = tok[27658:29113, :T].long()[:256]
print('  验证取 %s' % (tuple(val.shape),))


def masked_ce(x, mk, tag):
    """在给定掩码下算 CE + 掩码位 top-1 + 未掩码位 top-1"""
    xin = torch.where(mk, torch.full_like(x, MASK), x)
    with torch.no_grad():
        lg = model.forward_masked(xin, None)[..., :V0]
    if not bool(mk.any()):
        return
    ce = float(F.cross_entropy(lg[mk], x[mk]))
    acc_m = float((lg[mk].argmax(-1) == x[mk]).float().mean())
    unm = ~mk
    acc_u = float((lg[unm].argmax(-1) == x[unm]).float().mean()) if bool(unm.any()) else float('nan')
    print('  %-34s 掩码率 %5.1f%%  CE %7.4f  掩码位top-1 %6.2f%%  未掩码位top-1 %6.2f%%'
          % (tag, 100 * float(mk.float().mean()), ce, 100 * acc_m, 100 * acc_u))
    return ce


print()
print('[H1] 未掩码位置是否会"抄自己"')
mk50 = torch.rand(val.shape) < 0.5
masked_ce(val, mk50, '随机掩码 50%')
masked_ce(val, torch.zeros_like(val, dtype=torch.bool), '全不掩码（应≈随机 0.1%）')
print('  ⇒ 若"全不掩码"的 top-1 接近 100% ⇒ 模型在抄自己（H1 成立）')

print()
print('[H2] 用【新的随机掩码】重测（云端 val 用的是固定种子掩码）')
for r in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
    masked_ce(val, torch.rand(val.shape) < r, '新随机掩码 r=%.2f' % r)

print()
print('[H3] 结构性泄漏检查：改被掩码位的【真值】看 logits 变不变')
xin1 = torch.where(mk50, torch.full_like(val, MASK), val)
v2 = val.clone(); v2[mk50] = torch.randint(0, V0, (int(mk50.sum()),))
xin2 = torch.where(mk50, torch.full_like(v2, MASK), v2)
print('  两次输入是否完全相同: %s' % bool((xin1 == xin2).all()))
with torch.no_grad():
    a = model.forward_masked(xin1, None)[..., :V0]
    b = model.forward_masked(xin2, None)[..., :V0]
print('  掩码位 logits 最大差: %.3e  ⇒ %s'
      % ((a[mk50] - b[mk50]).abs().max().item(),
         'OK 无结构泄漏' if (a[mk50] - b[mk50]).abs().max() < 1e-6 else '❌ 有泄漏'))

print()
print('[H4] 云端 val 的固定掩码到底是怎样的')
_g = torch.Generator().manual_seed(20240918)
rr = torch.rand(val.shape[0], 1, generator=_g)
mkg = torch.rand(val.shape, generator=_g) < rr
print('  该掩码的总体掩码率 %.3f   每张图的 r: %s'
      % (float(mkg.float().mean()), [round(float(x), 3) for x in rr.flatten()[:8].tolist()]))
masked_ce(val, mkg, '云端 val 的固定掩码')
print()
print('=' * 92)
print('DONE')
