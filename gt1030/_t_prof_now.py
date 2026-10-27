# -*- coding: utf-8 -*-
"""当前（优化后）的时间构成：fwd_body / bwd_body / 整步 / AdamW / 传输。不挂起任何进程。"""
import sys
import time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_multi as MM

tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
model = MM.Model(r'F:\weibo_cos\ar_run_full3\ckpt_20000.pt')
model.forward(tokens); model.backward(tokens); model.opt_step(5e-6, 1)
d = model.dev


def t(fn, n=6):
    fn(); d.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    d.finish()
    return (time.time() - t0) / n * 1000


blk = model.blocks[0]
xh = np.zeros((MM.M, MM.MD), np.float16)
print('Block.fwd_body   %7.1f ms/层  => %6.0f ms/步' % (t(blk.fwd_body), t(blk.fwd_body) * 10))
print('Block.bwd_body   %7.1f ms/层  => %6.0f ms/步' % (t(blk.bwd_body), t(blk.bwd_body) * 10))
print('Model.forward    %7.1f ms' % t(lambda: model.forward(tokens)))
print('Model.backward   %7.1f ms' % t(lambda: model.backward(tokens)))
print('Model.opt_step   %7.1f ms' % t(lambda: model.opt_step(5e-6, 7), 3))
print('Model.grad_norm  %7.1f ms' % t(model.grad_norm, 3))
print('一整步           %7.1f ms'
      % t(lambda: (model.forward(tokens), model.backward(tokens), model.opt_step(5e-6, 9)), 4))
