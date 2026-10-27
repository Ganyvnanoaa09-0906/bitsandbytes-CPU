# -*- coding: utf-8 -*-
"""探针 4：在真实 V2ModelDev 构建过程中，逐层检查 sentinel 何时失效。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
d = MM.Dev()
sent = d.up(np.full(4096, 0.5, np.float16))


def check(tag):
    out = d.down(sent, (4096,))
    v = float(out[0])
    nz = int((out != 0.5).sum())
    print('%-22s sentinel[0]=%.3f  bad=%d' % (tag, v, nz))
    return v == 0.5


check('after Dev()')
orig_layer = MM.Layer.__init__


def layer_init(self, dev, sd, idx):
    orig_layer(self, dev, sd, idx)
    check('after Layer %d' % idx)


MM.Layer.__init__ = layer_init
model = MM.V2ModelDev(ckpt, dev=d)
check('after V2ModelDev')
print('layer0 Wqkv absmax=%.4f'
      % np.abs(d.down(model.layers[0].W['Wqkv'].w, (MM.MD, MM.WROW)).astype(np.float32)).max())
