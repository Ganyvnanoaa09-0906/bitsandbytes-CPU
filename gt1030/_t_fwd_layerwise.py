# -*- coding: utf-8 -*-
"""探针 34：逐层做 forward，打印每层输出的 absmax/mean，找出从哪层开始变垃圾。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_multi as MM

CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))

model = MM.Model(CKPT)
x = model.embed(tokens)
print('embed absmax=%.4f' % np.abs(x).max())
import torch.nn.functional as F
for i, blk in enumerate(model.blocks):
    x = blk.forward(x)
    a = x.astype(np.float32)
    print('block %d out: absmax=%.4f mean=%.5f nan=%d' % (i, np.abs(a).max(), a.mean(),
                                                          int(np.isnan(a).sum())))
model.head.forward(x)
l = model.head.loss(tokens)
print('loss=%.6f' % l)
lg = model.head.dev.down(model.head.logits, (MM.M, model.head.V), np.float16).astype(np.float32)
print('logits absmax=%.4f' % np.abs(lg).max())
