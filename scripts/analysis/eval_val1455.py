# -*- coding: utf-8 -*-
"""eval_val1455.py — 在【全部 1455 张】验证集上评估一个 ckpt，用来和 d61k 臂对齐口径。

背景（2026-09-19）：
  ar_flat256 的启动日志写的是 `train=27658 val=1455 (验证取固定 512 张)`
  ⇒ 它的 val 是【1455 张池子里的固定 512 张子集】的均值；
  而 ar_d61k_* 用的是 --val-range 27658:29113 ⇒ 全部 1455 张的均值。
  两者池子相同但【子集大小不同】⇒ 绝对值不可直接比。
  代码里那句"与 ar_flat256 验证的是同一批图 ⇒ val 可直接比"是【错的】。

用法:
  python eval_val1455.py --ar <ckpt.pt> --vqvae <ckpt.pt> --val-range 27658:29113
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
import argparse, os, sys, math, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', required=True)
    ap.add_argument('--vqvae', default='')
    ap.add_argument('--tokens', default=r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt')
    ap.add_argument('--val-range', default='27658:29113')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    ck = torch.load(a.ar, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    model = SmallARImageModelV2(cfg)

    sd = ck['model']
    if any(k.startswith('loop_mods.') for k in sd):
        from scripts.lib.loop_transformer import LoopConfig
        mx = 4
        for k, v in sd.items():
            if k.endswith('step_enc.table.weight') or k.endswith('cross_res.gate'):
                mx = v.shape[0]; break
        model.attach_loop(LoopConfig(max_steps=max(1, mx - 1)))
    r = model.load_state_dict(sd, strict=False)
    print('[eval] ckpt step=%s  missing=%d unexpected=%d' % (
        ck.get('step'), len(r.missing_keys), len(r.unexpected_keys)))
    print('[eval] use_bos=%s vocab=%s mask_mode=%s' % (
        cfg.use_bos, cfg.vocab_size, cfg.mask_mode))

    tok = torch.load(a.tokens, map_location='cpu', weights_only=False)
    T = cfg.total_tokens
    lo, hi = (int(x) for x in a.val_range.split(':'))
    va = torch.arange(lo, min(hi, tok.shape[0]))
    print('[eval] 验证集 = 行 [%d,%d)  共 %d 张  (tokens %s)' % (
        lo, min(hi, tok.shape[0]), len(va), tuple(tok.shape)))

    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(va), a.batch):
            sel = va[s:s + a.batch]
            if len(sel) == 0:
                continue
            xv = tok[sel, :T].long()
            (thc, thd), (gec, ged) = model.forward_logits(xv, None)
            if cfg.head_mode == 'factorized':
                from scripts.lib.small_image_model_v2 import FactorizedHead
                flat = model.merge(gec, ged, None)
            else:
                flat = model.merge(gec, ged, None)
            if cfg.use_bos:
                src, tgt = flat, xv
            else:
                src, tgt = flat[:, :-1], xv[:, 1:]
            l = F.cross_entropy(src.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
            tot += float(l) * len(sel); cnt += len(sel)

    ce = tot / max(cnt, 1)
    print('[eval] ★ 全 1455 张口径 CE = %.4f nats  (%.4f bits/token)' % (ce, ce / math.log(2)))
    print('[eval]   对照: 同 ckpt 在训练日志里的 val 是 512 张子集口径，两者不可直接比')


if __name__ == '__main__':
    main()
