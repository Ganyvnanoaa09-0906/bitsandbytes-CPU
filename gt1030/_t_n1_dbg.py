# -*- coding: utf-8 -*-
"""探针 23：N=1（单层整机）逐量对拍：host 嵌入 / device x_final / device logits vs torch。"""
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
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import torch.nn.functional as F
import gt1030.ar_v2_model as MM
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CKPT = r'F:\weibo_cos\ckpt_1l.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = SmallImageConfigV2(**ck['cfg'])
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'], strict=True)
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
xt = torch.from_numpy(tokens)

dev = MM.V2ModelDev(CKPT)
d = dev.dev
B, T, M, MD, V = MM.B, MM.T, MM.M, MM.MD, dev.V

def cmp(name, a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    if a.shape != b.shape:
        print('%-14s SHAPE %s vs %s' % (name, a.shape, b.shape)); return
    err = float(np.abs(a - b).max()); rel = err / (float(np.abs(b).max()) + 1e-12)
    print('%-14s max_abs=%.3e rel=%.3e devmax=%.4g refmax=%.4g %s'
          % (name, err, rel, float(np.abs(a).max()), float(np.abs(b).max()),
             'OK' if rel < 5e-2 else 'FAIL'))

with torch.no_grad():
    xe = m.embed(xt)                                   # (B,T,MD)
    xdev = dev.embed(tokens)
    cmp('embed', xdev, xe.view(M, MD).numpy())
    dev.forward(tokens)
    xf = d.down(dev.x_final, (M, MD), np.float16)
    lg = d.down(dev.sc['logits'], (M, V), np.float16)
    xb = m.blocks[0](xe)
    lgt = m.gen_head.lin(xb)[0] if isinstance(m.gen_head.lin(xb), tuple) else m.gen_head.lin(xb)
    cmp('x_final', xf, xb.view(M, MD).numpy())
    cmp('logits', lg, lgt.view(M, V).numpy())
    ref = F.cross_entropy(lgt[:, :-1].reshape(-1, V).float(), xt[:, 1:].reshape(-1))
print('torch loss = %.6f' % float(ref))
ld, ic = dev.loss_and_dlogits(tokens)
print('device loss = %.6f' % ld)
