"""eval_ar_ckpt.py -- pick the checkpoint by validation loss, not by training loss.

The run finished 3000/3000 steps with no error, but the per-step training loss is
noisy: it read 5.3649 at step 1801 and 5.8175 at step 3000, while each step sees
only 8 images. Choosing between those on a single batch's loss would be choosing on
noise, and sampling from the wrong checkpoint would make the whole comparison
meaningless.

train_ar_v2.py can record validation loss (--val_every/--val_frac) but neither was
set, so compute it here instead: hold out a fixed slice of the token file, evaluate
each checkpoint on it under no_grad, and report mean cross-entropy. The split uses
a fixed seed and is identical for every checkpoint, which is what makes the numbers
comparable.

Reported per checkpoint: mean validation loss, and the change from the previous one
so the plateau is visible.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--run', default=r'i5build\ar32_run1')
ap.add_argument('--tokens', default=r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt')
ap.add_argument('--n_val', type=int, default=512)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--threads', type=int, default=6)
a = ap.parse_args()

torch.set_num_threads(a.threads)
tok = torch.load(a.tokens, map_location='cpu')
if tok.ndim != 2:
    tok = tok.reshape(tok.shape[0], -1)
N, T = tok.shape
print('tokens: %d x %d' % (N, T))

# fixed split: the last n_val rows, identical for every checkpoint
val = tok[N - a.n_val:].long()
grid = int(round(T ** 0.5))

def cfg_from_ckpt(ck):
    """Rebuild the config the checkpoint was trained with.

    Building it by hand is what broke the first version of this script: the model
    uses GQA with n_kv_head=4, so k/v are (256,512), while the default n_kv_head=8
    gives (512,512) and load_state_dict fails on every block. The checkpoint stores
    its config, so use it rather than restating the fields.
    """
    import dataclasses
    saved = ck.get('cfg') or {}
    if not isinstance(saved, dict):
        saved = {}
    names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
    kw = {k: v for k, v in saved.items() if k in names}
    kw.setdefault('grid', grid)
    kw.setdefault('total_tokens', T)
    return SmallImageConfigV2(**kw)


print('grid=%d, 验证集 %d 条' % (grid, a.n_val))
print()

ckpts = sorted(glob.glob(os.path.join(a.run, 'ckpt_*.pt')),
               key=lambda p: int(re.search(r'ckpt_(\d+)', p).group(1))
               if re.search(r'ckpt_(\d+)', p) else 10 ** 9)
# ckpt_last.pt sorts oddly; put it at the end explicitly
last = os.path.join(a.run, 'ckpt_last.pt')
if os.path.exists(last) and last not in ckpts:
    ckpts.append(last)

print('%-16s %12s %10s %s' % ('ckpt', 'val loss', '相对上个', '步数'))
print('-' * 52)
prev = None
best = None
for p in ckpts:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck.get('model', ck)
    cfg = cfg_from_ckpt(ck)
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=False,
                          loop_cache='shared'))
    m.load_state_dict(sd, strict=False)
    m.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val), a.batch):
            x = val[i:i + a.batch]
            if x.shape[0] == 0:
                continue
            o = m.forward_logits(x, None)
            lg = o[1]
            lg = lg[0] if isinstance(lg, tuple) else lg
            # train_ar_v2 shifts the target; replicate that shift here
            lg2 = lg[:, :-1].reshape(-1, cfg.vocab_size)
            tg = x[:, 1:].reshape(-1)
            tot += float(torch.nn.functional.cross_entropy(lg2, tg, reduction='sum'))
            cnt += tg.numel()
    vl = tot / max(cnt, 1)
    step = ck.get('step', '?')
    tag = '' if prev is None else '%+.4f' % (vl - prev)
    print('%-16s %12.4f %10s %s' % (os.path.basename(p), vl, tag, step))
    if best is None or vl < best[0]:
        best = (vl, p)
    prev = vl
    del m, ck

print()
if best:
    print('  ⇒ 验证 loss 最低: %s（%.4f）' % (os.path.basename(best[1]), best[0]))
