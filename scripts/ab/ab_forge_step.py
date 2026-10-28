# -*- coding: utf-8 -*-
"""ab_forge_step.py —— 端到端 A/B：把 nn.Linear 换成 AVX2 内核，整步快多少。

★ 为什么不信算子剖析的占比：torch profiler 的 self_cpu_time_total 是【跨线程求和】的。
  一个 6 线程并行的算子，self time 会显示成约 6 倍墙上时间。所以「mul 占 26%」
  实际只有 2.9%（later 用 ab_rmsnorm.py 的墙钟对照证伪了）。
  唯一算数的是【同一进程、同一权重、同一批 token 的墙钟 min-of-N 对照】。

三档对照：
  torch         —— 原样
  forge-all     —— 所有 nn.Linear
  forge-ffn     —— 只换 w12 / w3（它们占 GEMM FLOPs 的 71%）

同时校验收敛前提：两档的 loss 和梯度必须一致（相对误差 < 1e-4），否则再快也不能用。
"""
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

from scripts.lib import forge_gemm
from scripts.lib import forge_patch
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)
FG.set_threads(TH)

CK = os.environ.get('CKPT', r'D:\work\bitsandbytes-CPU\.autopilot\v4_bos\ckpt_13000.pt')
B = int(os.environ.get('B', '8'))
REPS = int(os.environ.get('REPS', '5'))

print('=' * 94)
print('端到端 A/B：forge AVX2 内核 vs torch/oneDNN')
print('  线程=%d  B=%d  %s' % (TH, B, FG.info()))
print('=' * 94)

ck = torch.load(CK, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
sd = ck.get('model') or ck.get('state_dict') or ck
T = cfg.total_tokens
torch.manual_seed(1234)
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


def run(m, want_loss=False):
    out = m.forward_logits(tok)
    fs = flat(out)
    loss = sum(f.sum() for f in fs)
    loss.backward()
    if want_loss:
        return float(loss), [f.detach().clone() for f in fs]
    return float(loss), None


def grads_of(m):
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in m.named_parameters()}


def timeit(m, n=REPS):
    for _ in range(2):
        m.zero_grad(set_to_none=True)
        run(m)
    best = 1e30
    for _ in range(n):
        m.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        run(m)
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------- 数值一致性：先证明能换，再量快多少 ----------------
print()
print('---- ① 数值一致性（决定能不能换） ----')
m0 = build()
m0.zero_grad(set_to_none=True)
L0, F0 = run(m0, want_loss=True)
G0 = grads_of(m0)

for tag, only in (('forge-all', None), ('forge-ffn', ('w12', 'w3'))):
    m1 = build()
    names = forge_patch.enable(m1, only=only)
    m1.zero_grad(set_to_none=True)
    L1, F1 = run(m1, want_loss=True)
    G1 = grads_of(m1)
    dL = abs(L1 - L0) / max(abs(L0), 1e-30)
    # 梯度上界（只比参数，比的是残差范数 / 原范数）
    worst, worst_n = 0.0, ''
    for n in G0:
        if G0[n] is None or G1[n] is None:
            continue
        d = (G1[n] - G0[n]).norm().item()
        base = max(G0[n].norm().item(), 1e-30)
        if d / base > worst:
            worst, worst_n = d / base, n
    print('  %-10s 替换 %2d 个 Linear   loss 相对差 %.3e   梯度最大相对差 %.3e (%s)'
          % (tag, len(names), dL, worst, worst_n))
    forge_patch.disable(m1)

# ---------------- 墙钟 ----------------
print()
print('---- ② 墙钟（min-of-%d，同进程同数据） ----' % REPS)
res = {}
for tag, only in (('torch', None), ('forge-all', None), ('forge-ffn', ('w12', 'w3'))):
    m = build()
    names = []
    if tag != 'torch':
        names = forge_patch.enable(m, only=only)
    m.zero_grad(set_to_none=True)
    t = timeit(m)
    res[tag] = (t, len(names))
    forge_patch.disable(m)
    print('  %-10s %10.1f ms/步   %s' % (tag, t * 1e3, ('替换 %d 个' % len(names)) if names else ''))

base = res['torch'][0]
print()
print('  %-10s %12s %10s %14s' % ('版本', 'ms/步', '加速比', '1000 步省(分)'))
print('  ' + '-' * 52)
for tag in ('torch', 'forge-all', 'forge-ffn'):
    t = res[tag][0]
    print('  %-10s %12.1f %10.3fx %14.1f'
          % (tag, t * 1e3, base / t, (base - t) * 1000 / 60))
print()
print('DONE')
