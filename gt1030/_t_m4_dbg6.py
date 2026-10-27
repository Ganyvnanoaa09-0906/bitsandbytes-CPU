# -*- coding: utf-8 -*-
"""M4 debug 6: 构建后立刻检查多个 buffer，区分『全部为 0』还是『部分为 0』。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import pyopencl as cl
import gt1030.ar_v2_model as MM

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
ck = torch.load(ckpt, map_location='cpu', weights_only=False)
dev = MM.Dev()
q, ctx, mf = dev.q, dev.ctx, dev.mf

# 先证明基本通路
pat = np.full(1024, 0.5, np.float16)
pb = dev.up(pat)
print('pattern buffer readback: %.3f' % float(dev.down(pb, (1024,))[0]))

sd = ck['model']
ly = MM.Layer(dev, sd, 0)
print('layer0 Wqkv  : %.4f' % np.abs(dev.down(ly.W['Wqkv'].w, (MM.MD, MM.WROW)).astype(np.float32)).max())
print('layer0 W12   : %.4f' % np.abs(dev.down(ly.W['W12'].w, (MM.MD, 2 * MM.FF)).astype(np.float32)).max())
print('layer0 ln1   : %.4f' % np.abs(dev.down(ly.ln1, (MM.MD,)).astype(np.float32)).max())
print('layer0 ls1   : %.4f' % np.abs(dev.down(ly.ls1, (MM.MD,), np.float32)).max())

# 再把整机建起来，但用【同一个 Dev】
m2 = MM.V2ModelDev.__new__(MM.V2ModelDev)
MM.V2ModelDev.__init__(m2, ckpt, dev=dev)
print('after full build, layer0 Wqkv: %.4f'
      % np.abs(dev.down(m2.layers[0].W['Wqkv'].w, (MM.MD, MM.WROW)).astype(np.float32)).max())
print('after full build, layer5 Wqkv: %.4f'
      % np.abs(dev.down(m2.layers[5].W['Wqkv'].w, (MM.MD, MM.WROW)).astype(np.float32)).max())
print('after full build, head.w     : %.4f'
      % np.abs(dev.down(m2.head.w, (MM.MD, m2.V)).astype(np.float32)).max())
print('after full build, pattern    : %.3f' % float(dev.down(pb, (1024,))[0]))
n = 0
for ly2 in m2.layers:
    n += len(ly2.b) + 20
print('total buffers approx %d' % (n + len(m2.sc)))
