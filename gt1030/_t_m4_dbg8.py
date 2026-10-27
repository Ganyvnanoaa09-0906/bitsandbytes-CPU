# -*- coding: utf-8 -*-
"""M4 debug 8: 在 Param.__init__ 里立刻回读，定位是『上传就没上』还是『后来被清掉』。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM

orig_init = MM.Param.__init__
counter = {'n': 0}


def patched(self, dev, w_fp16, lr_scale=1.0):
    orig_init(self, dev, w_fp16, lr_scale)
    counter['n'] += 1
    if counter['n'] <= 4:
        back = dev.down(self.w, w_fp16.shape, np.float16).astype(np.float32)
        print('  Param#%d shape=%s  host_absmax=%.4f device_absmax=%.4f'
              % (counter['n'], w_fp16.shape, float(np.abs(w_fp16).max()),
                 float(np.abs(back).max())))


MM.Param.__init__ = patched

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
print('MM.B=%d MM.T=%d MM.M=%d  L.B=%d L.T=%d' % (MM.B, MM.T, MM.M, MM.L.B, MM.L.T))
model = MM.V2ModelDev(ckpt)
p = model.layers[0].W['Wqkv']
print('after build, layer0 Wqkv absmax=%.4f'
      % np.abs(model.dev.down(p.w, (MM.MD, MM.WROW)).astype(np.float32)).max())
print('after build, head absmax=%.4f'
      % np.abs(model.dev.down(model.head.w, (MM.MD, model.V)).astype(np.float32)).max())
