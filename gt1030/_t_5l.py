# -*- coding: utf-8 -*-
"""探针 20：用 5 层 ckpt（约 0.7GB 设备占用）跑前向 loss，看驱动是否稳定 + loss 是否对。

用法：python gt1030/_t_5l.py [ckpt]
"""
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
import gt1030.ar_v2_model as MM
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'F:\weibo_cos\ckpt_5l.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = SmallImageConfigV2(**ck['cfg'])
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'], strict=True)
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
xt = torch.from_numpy(tokens)

dev = MM.V2ModelDev(CKPT)
dev.forward(tokens)
loss_dev, ic = dev.loss_and_dlogits(tokens)
with torch.no_grad():
    lg = m.forward_logits(xt, None)[1][0]
import torch.nn.functional as F
loss_t = float(F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1)))
print('layers=%d  loss dev=%.6f torch=%.6f diff=%.2e'
      % (dev.n_layer, loss_dev, loss_t, abs(loss_dev - loss_t)))
