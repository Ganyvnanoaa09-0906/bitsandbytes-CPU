"""ab_muon_vs_adamw.py -- does Muon reach the same validation loss in fewer steps?

This is the only lever on the actual complaint. The 31-hour figure is steps x time-per-step;
time-per-step has at most 1.3x of room left (the matmuls already run at 79% of their
standalone rate, and the rest is attention and elementwise work), so the steps are where a
real reduction has to come from. And reducing steps without changing the loss target is not
a quality trade -- Muon's published claim is exactly that it gets there in fewer steps.

Two facts say this has never been tried here. train_ar_v2.py and train_ar_curriculum.py both
construct bnb.optim.AdamW8bit and nothing else. bench_muon.py measures only the cost of the
Newton-Schulz iteration, and its verdict thresholds use T=256, where forward+backward is
four times smaller than at T=1024 -- so its cost share is overstated for the configuration
actually used.

The hyperparameters are the ones recorded in bench_muon.py's header as supplied by the user:
lr 0.02, momentum 0.95 Nesterov, 5 NS steps, applied to 2-D weight matrices only, with
AdamW for embeddings and 1-D parameters.

Short sequences (T=256) so the comparison finishes in tens of minutes rather than hours. The
absolute losses will not match a T=1024 run and are not meant to: both arms get identical
settings, so the shape of the two curves and the step at which each reaches a given loss are
the comparison. Reported as loss at matched step counts and as steps-to-target.

Run as:  python ab_muon_vs_adamw.py [steps]
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import dataclasses
import math
import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402
import bitsandbytes as bnb  # noqa: E402

TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
torch.set_num_threads(6)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
T = 1024          # the real task shape; truncating the raster would change it
BATCH = 2         # the batch the real training uses
LR_ADAM = 1e-4
LR_MUON = 0.02
MOM = 0.95
# NS_ITERS lets the H3 experiment vary the Newton-Schulz iteration count without touching
# the default. os is imported later in this file, so import it inline here (the first attempt
# used a bare os.environ and died with NameError at import time).
NS_STEPS = int(__import__("os").environ.get("NS_ITERS", "5"))


def zeropower_via_newtonschulz5(G, steps=NS_STEPS, eps=1e-7):
    """Muon's Newton-Schulz orthogonalisation, from bench_muon.py (Keller Jordan)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2-D weights, Nesterov momentum, orthogonalised update."""

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
                d = p.grad.add(buf, alpha=g['momentum'])     # Nesterov
                if d.ndim == 2:
                    d = zeropower_via_newtonschulz5(d)
                    # scale so the update magnitude matches AdamW's regime
                    d = d * max(1.0, d.size(0) / d.size(1)) ** 0.5
                p.add_(d, alpha=-g['lr'])


tok = torch.load(TOKENS, map_location='cpu')


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        idx = torch.randint(0, len(tok), (BATCH,), generator=g)
        yield tok[idx].long()


def build():
    cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8,
                             n_kv_head=4, ffn_dim=2048, total_tokens=T,
                             grid=int(T ** 0.5), head_mode='flat', mask_mode='causal',
                             n_cluster=32, use_bos=False)
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=False,
                          loop_cache='shared'))
    return m, cfg


def make_opt(name, m):
    if name == 'adamw':
        return bnb.optim.AdamW8bit(m.parameters(), lr=LR_ADAM)
    # Muon on 2-D weight matrices, AdamW on everything else (embeddings, norms, 1-D)
    muon_p, adam_p = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (muon_p if (p.ndim == 2 and 'emb' not in n and 'head' not in n) else adam_p).append(p)
    # only the optimisers: returning the counts as well made the training loop try to
    # call zero_grad() on an int, which killed the run after the first arm had finished
    print('    Muon 覆盖 %d 个 2D 权重，AdamW 覆盖 %d 个其余参数' % (len(muon_p), len(adam_p)))
    return (Muon(muon_p, lr=LR_MUON), bnb.optim.AdamW8bit(adam_p, lr=LR_ADAM))


def run(name, steps, seed=0):
    m, cfg = build()
    m.train()
    opt = make_opt(name, m)
    data = batches(seed)
    losses, times = [], []
    for s in range(1, steps + 1):
        x = next(data)
        o = m.forward_logits(x, None)
        lg = o[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(
            lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        t0 = time.perf_counter()
        for o_ in (opt if isinstance(opt, tuple) else (opt,)):
            o_.zero_grad(set_to_none=True)
        loss.backward()
        for o_ in (opt if isinstance(opt, tuple) else (opt,)):
            o_.step()
        times.append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))
    return losses, statistics.median(times)


print('T=%d batch=%d，AdamW lr=%.0e vs Muon lr=%.3g（2D 权重）+ AdamW（其余）'
      % (T, BATCH, LR_ADAM, LR_MUON))
print('步数 %d；这是【曲线形状】的比较，绝对值不与 T=1024 的运行对应' % STEPS)
print()

import json
import os

RESULT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ab_muon_results.json')
res = {}
if os.path.exists(RESULT_JSON):
    try:
        with open(RESULT_JSON, encoding='utf-8') as f:
            res = json.load(f)
        print('  载入已有结果: %s' % list(res.keys()))
    except Exception:
        res = {}

for name in ('adamw', 'muon', 'muon3'):
    if name in res and len(res[name]) >= STEPS:
        print('  %-6s 已有 %d 步结果，跳过' % (name, len(res[name])))
        continue
    t0 = time.perf_counter()
    losses, opt_ms = run(name, STEPS)
    wall = time.perf_counter() - t0
    res[name] = losses
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f)
    print('  %-6s 跑完 %d 步，用时 %.1f s（优化器 %.1f ms/步）；结果已落盘'
          % (name, STEPS, wall, opt_ms * 1e3))

print()
print('%-8s %10s %10s %10s %10s' % ('步数', 'AdamW', 'Muon', '差值', '谁更好'))
for s in (50, 100, 200, 300, 400, 500, STEPS):
    if s > STEPS or 'adamw' not in res or 'muon' not in res:
        continue
    # running mean over the last 50 steps smooths single-batch noise
    w = 50
    a = statistics.mean(res['adamw'][max(0, s - w):s])
    b = statistics.mean(res['muon'][max(0, s - w):s])
    print('%-8d %10.4f %10.4f %+10.4f %10s'
          % (s, a, b, b - a, 'Muon' if b < a else 'AdamW'))

if 'adamw' not in res or 'muon' not in res:
    print()
    print('两臂尚未都有结果，无法比较。重跑本脚本会接着未完成的臂。')
    raise SystemExit(0)

print()
target = statistics.mean(res['adamw'][-50:])
for name in ('adamw', 'muon'):
    ls = res[name]
    hit = None
    w = 50
    for s in range(w, len(ls) + 1):
        if statistics.mean(ls[s - w:s]) <= target:
            hit = s
            break
    print('  达到 AdamW 终值 %.4f 所需步数: %-6s %s'
          % (target, name, hit if hit else '未达到'))
print()
print('判据: 若 Muon 用显著更少的步数达到同样的 loss ⇒ 那 31 小时可按该比例缩短；')
print('      若两者相当 ⇒ Muon 只增加优化器开销，应当放弃。')
