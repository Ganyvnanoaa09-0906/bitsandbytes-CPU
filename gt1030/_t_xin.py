# -*- coding: utf-8 -*-
"""探针 28：forward 之后直接看 L0.xin 的实际内容（判断是不是 tick 图案）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM

CKPT = r'F:\weibo_cos\ckpt_1l.pt'
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
dev = MM.V2ModelDev(CKPT)
d = dev.dev
x_host = dev.embed(tokens)
xa = d.down(dev.xa, (MM.M, MM.MD), np.float16).astype(np.float32)
print('xa      [:6]=%s absmax=%.4f' % (np.round(xa[0, :6], 3), np.abs(xa).max()))
dev.forward(tokens)
xin0 = d.down(dev.layers[0].b['xin'], (MM.M, MM.MD), np.float16).astype(np.float32)
print('L0.xin  [:6]=%s absmax=%.4f  与 xa 相同? %s'
      % (np.round(xin0[0, :6], 3), np.abs(xin0).max(), np.allclose(xin0, xa, atol=1e-3)))
print('host    [:6]=%s' % np.round(x_host[0, :6], 3))
h1 = d.down(dev.layers[0].b['h1'], (MM.M, MM.MD), np.float16).astype(np.float32)
print('L0.h1   [:6]=%s absmax=%.4f' % (np.round(h1[0, :6], 3), np.abs(h1).max()))
lg = d.down(dev.sc['logits'], (MM.M, dev.V), np.float16).astype(np.float32)
print('logits  [0,:6]=%s absmax=%.4f' % (np.round(lg[0, :6], 3), np.abs(lg).max()))
