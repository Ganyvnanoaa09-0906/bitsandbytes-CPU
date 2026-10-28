# -*- coding: utf-8 -*-
"""M5 自测：加载 ckpt -> 导出 -> 与原始 ckpt 对比（应只差 fp16 量化误差）-> 能被 torch 模型加载。

    python gt1030/ar_v2_export_test.py [--steps N]
"""
from __future__ import annotations
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
import argparse
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch

import gt1030.ar_v2_multi as MM
import gt1030.ar_v2_export as EX
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
TOKENS = r'F:\weibo_cos\tokens_32x32_weibo.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--out', default=r'F:\weibo_cos\ar_run_igpu\ckpt_igpu_test.pt')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--lr', type=float, default=5e-5)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    tok = torch.load(TOKENS, map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))

    model = MM.Model(args.ckpt)
    t0 = time.time()
    n_steps = args.steps
    loss0 = model.forward(tokens)
    model.backward(tokens)
    print('初始 loss=%.6f' % loss0)
    for st in range(1, n_steps + 1):
        l = model.forward(tokens)
        model.backward(tokens)
        gn = model.opt_step(args.lr, st)
        print('  step %d loss=%.6f gnorm=%.4f' % (st, l, gn))
    print('训练 %.2fs' % (time.time() - t0))

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    EX.export_ckpt(model, args.out, step=ck.get('step', 0) + n_steps)
    print('导出 %s' % args.out)

    new = torch.load(args.out, map_location='cpu', weights_only=False)
    print('--- 与原始 ckpt 对比（fp16 往返误差）---')
    rows = EX.ckpt_diff(ck, new)
    bad = [r for r in rows if isinstance(r[2], float) and r[2] > 5e-3]
    for k, e, rel in rows[:4]:
        print('  %-28s %s rel=%s' % (k, e, ('%.3e' % rel) if isinstance(rel, float) else rel))
    print('  共 %d 项, rel>5e-3 的 %d 项' % (len(rows), len(bad)))
    for k, e, rel in bad[:6]:
        print('   ! %-28s %s rel=%.3e' % (k, e, rel))

    # torch 侧能否加载 + 前向一致
    cfg = SmallImageConfigV2(**new['cfg'])
    m2 = SmallARImageModelV2(cfg)
    m2.load_state_dict(new['model'], strict=True)
    print('torch 模型 load_state_dict: OK (%d 层)' % cfg.n_layer)
    xt = torch.from_numpy(tokens)
    with torch.no_grad():
        lg = m2.forward_logits(xt, None)[1][0]
    import torch.nn.functional as F
    l_t = float(F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1)))
    l_d = model.forward(tokens)
    print('导出的 ckpt 前向 loss: torch=%.6f device=%.6f 差=%.2e'
          % (l_t, l_d, abs(l_t - l_d)))


if __name__ == '__main__':
    main()
