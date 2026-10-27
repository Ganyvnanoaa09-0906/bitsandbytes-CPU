# -*- coding: utf-8 -*-
"""探针 25：多层（N=3）逐层对拍 b['xin'] 与 x_final，找出第几层开始错。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'F:\weibo_cos\ckpt_3l.pt'
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
M, MD = MM.M, MM.MD
dev.forward(tokens)

with torch.no_grad():
    x = m.embed(xt)
    for i, blk in enumerate(m.blocks):
        din = d.down(dev.layers[i].b['xin'], (M, MD), np.float16).astype(np.float32)
        ref_in = x.view(M, MD).numpy()
        err = np.abs(din - ref_in).max()
        print('layer %d input : max_abs=%.3e (refmax=%.3f) %s'
              % (i, err, np.abs(ref_in).max(), 'OK' if err < 5e-2 else 'FAIL'))
        x = blk(x)
    xf = d.down(dev.x_final, (M, MD), np.float16).astype(np.float32)
    print('x_final       : max_abs=%.3e' % np.abs(xf - x.view(M, MD).numpy()).max())
    lg = d.down(dev.sc['logits'], (M, dev.V), np.float16).astype(np.float32)
    lgt = m.gen_head.lin(x)
    lgt = lgt[0] if isinstance(lgt, (tuple, list)) else lgt
    print('logits        : max_abs=%.3e (refmax=%.3f)'
          % (np.abs(lg - lgt.view(M, -1).numpy()).max(), float(lgt.abs().max())))
