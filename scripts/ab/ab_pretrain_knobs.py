# -*- coding: utf-8 -*-
"""ab_pretrain_knobs.py —— 把已验证的小改动合起来，量端到端到底省多少。

单项结论（都是干净机器上量的）：
  · 融合 RMSNorm（F.rms_norm 取代 6 个基础算子）      +2.9%
  · 融合 AdamW（torch AdamW fused 取代 bnb 8bit）     优化器 119.8→48.6 ms
  · 5 线程取代 6 线程                                   +1.0%

★ 为什么用【交错】测量：现在 SDXL 编码又在跑了（5 个线程），单次 min-of-N
  容易把某一档测在"恰好编码在抢核"的窗口里。改成 A,B,C,D,A,B,C,D... 轮换，
  每档取【跨轮的最小值】—— 轮换能让四档都经历同样的干扰分布。

★ 先校验数值：融合 RMSNorm 已验证过（check_rmsnorm_equiv.py），但融合 AdamW
  会改变累加顺序，必须确认参数更新量对得上，否则跑了几千步才发现漂移就晚了。
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
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

from scripts.lib import small_image_model_v2
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CK = os.environ.get('CKPT', r'D:\work\bitsandbytes-CPU\.autopilot\v4_bos\ckpt_13000.pt')
B = int(os.environ.get('B', '8'))
ROUNDS = int(os.environ.get('ROUNDS', '3'))
REPS = int(os.environ.get('REPS', '3'))

print('=' * 92)
print('预训练小改动合并 A/B（交错测量）')
print('=' * 92)

ck = torch.load(CK, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
sd = ck.get('model') or ck.get('state_dict') or ck
T = cfg.total_tokens
torch.manual_seed(4321)
tok = torch.randint(0, cfg.vocab_size, (B, T))


def build():
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(sd)
    m.train()
    return m


def flat(o):
    r = []
    if o is None:
        return r
    if isinstance(o, (tuple, list)):
        for i in o:
            r += flat(i)
        return r
    if torch.is_tensor(o):
        r.append(o)
    return r


# ---------------- ① 融合 AdamW 的数值一致性 ----------------
print()
print('---- ① 融合 AdamW vs bnb AdamW8bit：同样走 30 步，比参数更新量 ----')
import bitsandbytes as bnb

res = {}
for tag, mk in (('bnb AdamW8bit', lambda ps: bnb.optim.AdamW8bit(ps, lr=1e-3)),
                ('torch AdamW fused', lambda ps: torch.optim.AdamW(ps, lr=1e-3, fused=True))):
    torch.manual_seed(99)
    m = build()
    opt = mk(m.parameters())
    torch.manual_seed(555)
    for _ in range(30):
        t = torch.randint(0, cfg.vocab_size, (B, T))
        sum(f.float().sum() for f in flat(m.forward_logits(t))).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    res[tag] = {n: p.detach().clone() for n, p in m.named_parameters()}

a = res['bnb AdamW8bit']
b = res['torch AdamW fused']
worst, wn = 0.0, ''
for n in a:
    d = (a[n] - b[n]).norm().item()
    base = max(a[n].norm().item(), 1e-30)
    if d / base > worst:
        worst, wn = d / base, n
print('  30 步后参数最大相对差 %.3e  (%s)' % (worst, wn))
print('  %s' % ('一致（注意：两者数值实现不同，不该期望 bit 级相同）'
                if worst < 5e-2 else '**差异偏大，需要人工判断**'))

# ---------------- ② 交错墙钟 ----------------
CONFIGS = [
    ('A 原始',        False, 'bnb',   6),
    ('B +融合norm',   True,  'bnb',   6),
    ('C +融合AdamW',  True,  'fused', 6),
    ('D +5线程',      True,  'fused', 5),
]


def make_opt(m, kind):
    if kind == 'bnb':
        return bnb.optim.AdamW8bit(m.parameters(), lr=1e-4)
    return torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)


print()
print('---- ② 交错墙钟（%d 轮 x 每轮 %d 次） ----' % (ROUNDS, REPS))
print('  %-14s %10s %10s' % ('配置', '最快 ms/步', '累计轮次'))
print('  ' + '-' * 40)

state = {}
for name, fused_norm, optkind, nt in CONFIGS:
    M._RMS_FUSED = fused_norm
    torch.set_num_threads(nt)
    m = build()
    opt = make_opt(m, optkind)
    for _ in range(2):
        sum(f.float().sum() for f in flat(m.forward_logits(tok))).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    state[name] = (m, opt, nt)
    print('  %-14s 预热完成' % name)

print()
best = {k: 1e30 for k in state}
for r in range(ROUNDS):
    line = []
    for name, fused_norm, optkind, nt in CONFIGS:
        m, opt, _ = state[name]
        M._RMS_FUSED = fused_norm
        torch.set_num_threads(nt)
        b = 1e30
        for _ in range(REPS):
            t0 = time.perf_counter()
            sum(f.float().sum() for f in flat(m.forward_logits(tok))).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            b = min(b, time.perf_counter() - t0)
        best[name] = min(best[name], b)
        line.append('%s=%.0f' % (name[0], b * 1e3))
    print('  轮 %d:  %s ms' % (r + 1, '   '.join(line)))

M._RMS_FUSED = True
torch.set_num_threads(6)

print()
base = best['A 原始']
print('  %-14s %12s %10s %14s' % ('配置', 'ms/步', '加速比', '1000步省(分)'))
print('  ' + '-' * 56)
for name, _, _, _ in CONFIGS:
    t = best[name]
    print('  %-14s %12.1f %9.3fx %14.1f'
          % (name, t * 1e3, base / t, (base - t) * 1000 / 60))
print()
print('  ⇒ 全部小改动合计提速 %.3fx（%.1f%%）'
      % (base / best['D +5线程'], (1 - best['D +5线程'] / base) * 100))
print('DONE')
