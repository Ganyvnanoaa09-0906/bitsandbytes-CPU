# -*- coding: utf-8 -*-
"""M4 debug 2: where do the zeros start?"""
import sys, time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM
import gt1030.full_gpu_layer as L

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
d = MM.V2ModelDev(ckpt)
dev = d.dev

def stat(name, buf, shape):
    a = dev.down(buf, shape, np.float16).astype(np.float32)
    print('%-12s absmax=%.4f  mean=%.6f  nan=%s' % (name, np.abs(a).max(), a.mean(),
                                                    int(np.isnan(a).sum())))
    return a

stat('Wqkv[0]', d.layers[0].W['Wqkv'].w, (MM.MD, MM.WROW))
stat('W12[0]', d.layers[0].W['W12'].w, (MM.MD, 2 * MM.FF))
stat('head.w', d.head.w, (MM.MD, d.V))
x = d.embed(tokens)
print('embed x: absmax=%.4f' % np.abs(x).max())
d.forward(tokens)
stat('h1[0]', d.layers[0].b['h1'], (MM.M, MM.MD))
stat('attn[0]', d.sc['attn'], (MM.M, MM.MD))
stat('x1[0]', d.layers[0].b['x1'], (MM.M, MM.MD))
stat('a2[0]', d.layers[0].b['a2'], (MM.M, 2 * MM.FF))
stat('g[0]', d.layers[0].b['g'], (MM.M, MM.FF))
stat('xfinal', d.x_final, (MM.M, MM.MD))
stat('logits', d.sc['logits'], (MM.M, d.V))
