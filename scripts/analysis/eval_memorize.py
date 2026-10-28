"""eval_memorize.py -- how well does a checkpoint reproduce the sequences it was trained on?

The overfit experiments ask a binary question ("did it memorise?"), and a single terminal loss is a
bad answer: a model can hold CE ~2.0 while its greedy replay is visibly wrong, and it can look right
on a loss curve while free sampling collapses. This script reports the three measurements that
actually distinguish the answers, for any tokens file:

  A. shifted teacher-forced CE on the rows (the trainer's own convention: logits[:, i] predicts
     tokens[:, i + 1]);
  B. greedy reconstruction: first token from the truth, every later token argmax'd from the
     teacher-forced logits -- decoded to TRUE / TF image grids so the answer is visual;
  C. free samples from the loaded checkpoint (same sampler as the runs), decoded to a FREE grid.

Usage:
  python -u eval_memorize.py --ar <ckpt> --vqvae <ckpt> --tokens <tokens.pt> \
      --n 8 --start 0 --out-prefix i5build/eval/overfit_single
"""
from __future__ import annotations
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

import argparse
import os
import sys

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.probe.probe_exposure import load_ar, topk_sample, decode_grid


def gen_logits(model, seq):
    with torch.no_grad():
        _think, gen = model.forward_logits(seq, None)
    if isinstance(gen, (tuple, list)):
        cl, cd = gen
        return model.merge(cl, cd, None) if hasattr(model, 'merge') else cl
    return gen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', required=True)
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--no-samples', action='store_true', help='skip free sampling')
    args = ap.parse_args()

    model, cfg = load_ar(args.ar)
    T = cfg.total_tokens
    side = int(round(T ** 0.5))
    tokens = torch.load(args.tokens, map_location='cpu', weights_only=False).long()
    n = min(args.n, tokens.shape[0])
    true = tokens[args.start:args.start + n]
    print('cfg: T=%d grid=%d use_bos=%s vocab=%d' % (T, side, getattr(cfg, 'use_bos', None), cfg.vocab_size))
    print('rows: %d from %d' % (n, args.start))

    # A. CE, convention depends on use_bos: use_bos=True already shifted by BOS, so logits[:, i]
    # predicts tokens[:, i]; use_bos=False predicts tokens[:, i+1] and needs a shift here.
    lg = gen_logits(model, true)
    use_bos = bool(getattr(cfg, 'use_bos', False))
    if use_bos:
        ce = float(F.cross_entropy(lg.reshape(-1, cfg.vocab_size), true.reshape(-1)))
        label = 'unshifted (use_bos=True)'
    else:
        ce = float(F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size), true[:, 1:].reshape(-1)))
        label = 'shifted (use_bos=False)'
    print('[A] CE on these rows [%s]: %.4f nats  (random %.3f)'
          % (label, ce, torch.log(torch.tensor(float(cfg.vocab_size)))))

    # B. greedy reconstruction (same convention as A)
    if use_bos:
        tf = lg.argmax(-1)
    else:
        tf = true.clone()
        tf[:, 1:] = lg[:, :-1].argmax(-1)
    agree = float((tf == true).float().mean())
    print('[B] greedy reconstruction positional agreement: %.2f%%' % (agree * 100))
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    decode_grid(args.vqvae, true, side, args.out_prefix + '_TRUE.png')
    decode_grid(args.vqvae, tf, side, args.out_prefix + '_TF.png')
    print('    wrote %s_{TRUE,TF}.png' % args.out_prefix)

    # C. free samples
    if not args.no_samples:
        with torch.no_grad():
            free = model.generate(None, steps=T, top_k=100, temperature=1.0, seed=0, batch=n)
        decode_grid(args.vqvae, free, side, args.out_prefix + '_FREE.png')
        print('    wrote %s_FREE.png' % args.out_prefix)
    return 0


if __name__ == '__main__':
    sys.exit(main())
