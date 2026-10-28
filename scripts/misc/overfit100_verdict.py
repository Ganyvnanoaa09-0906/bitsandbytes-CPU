"""Verdict for the overfit-100 run: can the model FIT the 100 pictures, and is free sampling the gap?

The run finished 2000 steps (20 epochs) with train loss ~2.0 while free samples are still texture. That
is not yet a verdict, because three different things can produce it:

  A. the model cannot fit at all         -> train CE high under teacher forcing
  B. it fits, but free sampling collapses -> train CE low, greedy reconstruction close to the true
                                            tokens, prefix-conditioned continuation coherent
  C. it fits and free samples are just bad luck -> free sample would look like a training picture

So measure all of them on the same checkpoint and the same 8 training rows:
  * shifted CE (the convention the trainer uses: logits[:, i] predicts tokens[:, i+1])
  * greedy teacher-forced reconstruction: first token kept, the rest predicted from the true prefix
  * prefix-conditioned continuation: 896 true tokens, 128 generated, then compared token by token
  * free samples from the finished run (already on disk)

Saves TRUE / TF / PREFIX image grids for visual comparison.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import os
import sys
import time

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.probe.probe_exposure import load_ar, topk_sample, decode_grid

CKPT = r'D:\work\bitsandbytes-CPU\i5build\ar_overfit100_v3\ckpt_last.pt'
VQ = r'D:\work\bitsandbytes-CPU\i5build\vqgan32\ckpt_last.pt'
TOK = r'D:\work\bitsandbytes-CPU\tokens_overfit100.pt'
OUT = r'D:\work\research\evidence\overfit100_verdict'
os.makedirs(OUT, exist_ok=True)

model, cfg = load_ar(CKPT)
T = cfg.total_tokens
side = int(round(T ** 0.5))
tokens = torch.load(TOK, map_location='cpu', weights_only=False).long()
print('cfg T=%d grid=%d use_bos=%s vocab=%d' % (T, side, getattr(cfg, 'use_bos', None), cfg.vocab_size))
print('tokens', tuple(tokens.shape))


def gen_logits(seq):
    with torch.no_grad():
        _think, gen = model.forward_logits(seq, None)
    if isinstance(gen, (tuple, list)):
        cl, cd = gen
        return model.merge(cl, cd, None) if hasattr(model, 'merge') else cl
    return gen


# ---- A. shifted CE on training rows (the trainer's convention) ----
n = 16
true = tokens[:n]
lg = gen_logits(true)
ce = float(F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size), true[:, 1:].reshape(-1)))
print()
print('[A] shifted CE on 16 training rows: %.3f nats  (random = %.3f)' % (ce, torch.log(torch.tensor(float(cfg.vocab_size)))))
print('    => %s' % ('FITS (CE well below random)' if ce < 4.0 else 'DOES NOT FIT under teacher forcing'))

# ---- B. greedy teacher-forced reconstruction ----
tf = true.clone()
tf[:, 1:] = lg[:, :-1].argmax(-1)
agree = float((tf == true).float().mean())
pos_agree = (tf == true).float().mean(0)
print()
print('[B] greedy TF reconstruction: positional agreement %.1f%%  (first token forced)'
      % (agree * 100))
print('    first 25%%: %.1f%%   last 25%%: %.1f%%'
      % (100 * float(pos_agree[:T // 4].mean()), 100 * float(pos_agree[3 * T // 4:].mean())))
decode_grid(VQ, true[:4], side, os.path.join(OUT, 'TRUE.png'))
decode_grid(VQ, tf[:4], side, os.path.join(OUT, 'TF_GREEDY.png'))

# ---- C. prefix-conditioned continuation: 896 true tokens, 128 generated ----
prefix = 896
m = 2
seq = true[:m, :prefix].clone()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(T - prefix):
        lg1 = gen_logits(seq)
        nxt = topk_sample(lg1[:, -1], 100, 1.0)
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
print()
print('[C] prefix-conditioned: %d true + %d generated in %.1f s'
      % (prefix, T - prefix, time.perf_counter() - t0))
cont_true = true[:m, prefix:]
cont_gen = seq[:, prefix:]
agree_c = float((cont_gen == cont_true).float().mean())
print('    continuation token agreement vs truth: %.1f%%  (chance = %.2f%%)'
      % (agree_c * 100, 100.0 / cfg.vocab_size))
decode_grid(VQ, true[:m], side, os.path.join(OUT, 'PREFIX_TRUE.png'))
decode_grid(VQ, seq, side, os.path.join(OUT, 'PREFIX_GEN.png'))

print()
print('outputs in', OUT)
