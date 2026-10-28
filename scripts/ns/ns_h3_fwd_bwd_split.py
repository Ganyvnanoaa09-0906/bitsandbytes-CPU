"""Set up the forward/backward decomposition for this A/B harness, to be run once the
2000-step arm finishes.

Why it is needed: the microbenchmark gives an optimiser-stage saving (4443 -> 3537 ms/step,
20.4%) and the plan explicitly forbids quoting that as an end-to-end speedup. Converting it
requires knowing how much of a step the optimiser actually is, which means measuring the
forward and backward in this harness. That measurement has not been made, so the honest state
is that the 20.4% has no whole-step equivalent yet.

Written now but NOT run now: it would compete with the running training for the same six
cores, and contaminating the very run it is meant to interpret would be the same mistake
recorded in 7.26.2. It runs after.

Scope limit, stated up front so the number is not over-read: this measures the HARNESS's step
(batch 2, T=1024, the same model, no validation inside the loop). It is not the trainer's
3.952 s/step, which includes verification every 250 steps. Both are legitimate; they are not
interchangeable, and the report keeps them in separate rows.
"""
from __future__ import annotations
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

import io
import os
import statistics
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


CHECK = os.path.join(r'D:\work\bitsandbytes-CPU', 'ns_ab_muon3_2000.log')
done = False
try:
    t = io.open(CHECK, encoding='utf-8', errors='replace').read()
    done = 'muon3' in t and '跑完' in t
except Exception:
    pass

print('  muon3 2000 步是否结束: %s' % ('是' if done else '否'))
if not done:
    print()
    print('  ⇒ 现在【不运行】本分解测量：它会和正在跑的训练抢同一批 6 个核，')
    print('    而污染它要解释的那次运行，正是 7.26.2 记录过的错误。')
    print('    等 muon3 落盘后再跑本脚本即可。')
    sys.exit(0)

import torch
import bitsandbytes as bnb
from scripts.lib.measure_guard import assert_idle, stamp
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

assert_idle()
torch.set_num_threads(6)

TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
T, B = 1024, 2


def ns(G, steps, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    return X.T if tr else X


cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
                         ffn_dim=2048, total_tokens=T, grid=32, head_mode='flat',
                         mask_mode='causal', n_cluster=32, use_bos=False)
tok = torch.load(TOK, map_location='cpu')

stamp('H3：harness 的前向/反向分解')
print()
res = {}
for label, ns_iters in (('NS=5', 5), ('NS=3', 3)):
    m = SmallARImageModelV2(cfg)
    m.train()
    mp, ap = [], []
    for n_, p in m.named_parameters():
        if p.requires_grad:
            (mp if (p.ndim == 2 and 'emb' not in n_ and 'head' not in n_) else ap).append(p)
    aopt = bnb.optim.AdamW8bit(ap, lr=1e-4)
    buf = [torch.zeros_like(p) for p in mp]
    g = torch.Generator().manual_seed(0)
    fw, bw, op = [], [], []
    for _ in range(20):
        x = tok[torch.randint(0, len(tok), (B,), generator=g)].long()
        t0 = time.perf_counter()
        out = m.forward_logits(x, None)
        lg = out[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(
            lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        t1 = time.perf_counter()
        aopt.zero_grad(set_to_none=True)
        loss.backward()
        t2 = time.perf_counter()
        # p.add_ needs no_grad: p is a leaf requiring grad, so an in-place update outside
        # no_grad raises. The first version of this script died here and the 2000-step run
        # finished before anyone noticed -- the guard had never let it run before.
        with torch.no_grad():
            for k, p in enumerate(mp):
                buf[k].mul_(0.95).add_(p.grad)
                d = p.grad.add(buf[k], alpha=0.95)
                d = ns(d, ns_iters) * max(1.0, d.size(0) / d.size(1)) ** 0.5
                p.add_(d, alpha=-0.02)
        aopt.step()
        t3 = time.perf_counter()
        fw.append(t1 - t0)
        bw.append(t2 - t1)
        op.append(t3 - t2)
    f, b_, o = (statistics.median(fw), statistics.median(bw), statistics.median(op))
    res[label] = (f, b_, o)
    print('  %s  前向 %.3f s  反向 %.3f s  优化器 %.3f s  合计 %.3f s' % (label, f, b_, o, f + b_ + o))

f5, b5, o5 = res['NS=5']
f3, b3, o3 = res['NS=3']
tot5, tot3 = f5 + b5 + o5, f3 + b3 + o3
print()
print('  口径③：本 harness 的完整一步')
print('    NS=5 ⇒ %.3f s     NS=3 ⇒ %.3f s     加速 %.4f×' % (tot5, tot3, tot5 / tot3))
print('    其中优化器从 %.3f s 降到 %.3f s（省 %.0f ms，占整步 %.2f%%）'
      % (o5, o3, (o5 - o3) * 1000, (o5 - o3) / tot5 * 100))
print()
print('  两项声明:')
print('    1. 这是【本 harness】的一步，不含每 250 步的验证；训练器的 3.952 s/步 是另一口径。')
print('    2. 优化器占比 %.1f%% 决定了 NS 节省能传导多少到整步，这个比例不能外推到训练器。'
      % (o5 / tot5 * 100))
