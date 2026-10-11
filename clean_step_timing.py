"""clean_step_timing.py -- the seconds-per-step each arm actually costs, measured idle.

The 2000-step A/B ran both arms, but the Muon arm ran while I was doing iGPU experiments,
so its per-step cost is inflated by an unknown amount: 5.94 s against the 5.09 s the
600-step run measured under cleaner conditions. AdamW's arm was undisturbed at 3.32 s.
Section 7.26.2 records how that happened.

That number decides the headline. Step-count ratio is 3.19x, but the time ratio is
3.19 / (s_muon / s_adamw), which is 1.78x at 5.94 s and 2.08x at 5.09 s -- a 17% swing in
the answer, which is too much to leave to a guess.

So: 100 steps each, machine idle, guard active. Short because only the per-step cost is
being measured here, not convergence -- the loss curves already exist and are unaffected by
this.

Reported per arm: median seconds per step, the forward/backward split, and the resulting
time-to-loss ratio recomputed with the clean figure.
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from measure_guard import assert_idle, stamp   # noqa: E402
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402
import bitsandbytes as bnb  # noqa: E402

assert_idle()
stamp('干净计时开始')
print()

TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
torch.set_num_threads(6)
T, BATCH = 1024, 2
LR_ADAM, LR_MUON, MOM, NS_STEPS = 1e-4, 0.02, 0.95, 5
STEPS = 100


def zeropower_via_newtonschulz5(G, steps=NS_STEPS, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    tr = False
    if X.size(0) > X.size(1):
        X = X.T
        tr = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tr else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=LR_MUON, momentum=MOM):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                buf = self.state[p].get('buf')
                if buf is None:
                    buf = torch.zeros_like(p)
                    self.state[p]['buf'] = buf
                buf.mul_(g['momentum']).add_(p.grad)
                d = p.grad.add(buf, alpha=g['momentum'])
                if d.ndim == 2:
                    d = zeropower_via_newtonschulz5(d)
                    d = d * max(1.0, d.size(0) / d.size(1)) ** 0.5
                p.add_(d, alpha=-g['lr'])


tok = torch.load(TOKENS, map_location='cpu')


def build():
    cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8,
                             n_kv_head=4, ffn_dim=2048, total_tokens=T,
                             grid=int(T ** 0.5), head_mode='flat', mask_mode='causal',
                             n_cluster=32, use_bos=False)
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=False,
                          loop_cache='shared'))
    m.train()
    return m, cfg


def make_opt(name, m):
    if name == 'adamw':
        return bnb.optim.AdamW8bit(m.parameters(), lr=LR_ADAM)
    muon_p, adam_p = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (muon_p if (p.ndim == 2 and 'emb' not in n and 'head' not in n) else adam_p).append(p)
    return Muon(muon_p, lr=LR_MUON), bnb.optim.AdamW8bit(adam_p, lr=LR_ADAM)


def measure(name):
    m, cfg = build()
    opt = make_opt(name, m)
    opts = opt if isinstance(opt, tuple) else (opt,)
    g = torch.Generator().manual_seed(0)
    fwd, bwd, optt = [], [], []
    for _ in range(STEPS):
        x = tok[torch.randint(0, len(tok), (BATCH,), generator=g)].long()
        t0 = time.perf_counter()
        o = m.forward_logits(x, None)
        lg = o[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(
            lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        t1 = time.perf_counter()
        for o_ in opts:
            o_.zero_grad(set_to_none=True)
        loss.backward()
        t2 = time.perf_counter()
        for o_ in opts:
            o_.step()
        t3 = time.perf_counter()
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
        optt.append(t3 - t2)
    return (statistics.median(fwd), statistics.median(bwd), statistics.median(optt))


print('%d 步，batch %d，T=%d，空闲机器' % (STEPS, BATCH, T))
print()
print('%-8s %10s %10s %10s %10s' % ('臂', '前向 s', '反向 s', '优化器 s', '合计 s'))
print('-' * 54)
res = {}
for name in ('adamw', 'muon'):
    f, b, o = measure(name)
    res[name] = f + b + o
    print('%-8s %10.4f %10.4f %10.4f %10.4f' % (name, f, b, o, f + b + o))

print()
ratio_step = 3.19     # from the 2000-step A/B
speed = res['muon'] / res['adamw']
print('Muon 每步是 AdamW 的 %.3f×' % speed)
print('⇒ 步数比 3.19× ⇒ 【时间比 = 3.19 / %.3f = %.2f×】' % (speed, ratio_step / speed))
print()
try:
    with open(r'D:\work\bitsandbytes-CPU\ab_muon_results.json', encoding='utf-8') as fh:
        j = json.load(fh)
    if 'adamw' in j and 'muon' in j:
        print('（对照：被污染的 2000 步 A/B 给出 Muon %.3f s/步、AdamW %.3f s/步）'
              % (11883.8 / 2000, 6631.6 / 2000))
except Exception:
    pass
