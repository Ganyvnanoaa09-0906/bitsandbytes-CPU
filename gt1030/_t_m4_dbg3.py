# -*- coding: utf-8 -*-
"""M4 debug 3: isolate why V2ModelDev's weights look zero."""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM

ck = torch.load(r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt', map_location='cpu', weights_only=False)
sd = ck['model']
W = torch.cat([sd['blocks.0.attn.q.weight'], sd['blocks.0.attn.k.weight'],
               sd['blocks.0.attn.v.weight']], 0).T.contiguous().numpy().astype(np.float16)
print('host W absmax', np.abs(W).max(), W.shape, W.flags['C_CONTIGUOUS'])

d = MM.V2ModelDev(r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt')
p = d.layers[0].W['Wqkv']
print('after V2ModelDev: n=%d  w absmax=%.4f'
      % (p.n, np.abs(d.dev.down(p.w, (MM.MD, MM.WROW)).astype(np.float32)).max()))
print('master absmax=%.4f'
      % np.abs(d.dev.down(p.master, (p.n,), np.float32)).max())
print('shape of p.w buffer bytes? n*2 =', p.n * 2)
# re-upload manually into the same buffer object
import pyopencl as cl
cl.enqueue_copy(d.dev.q, p.w, W)
d.dev.finish()
print('after manual re-upload: absmax=%.4f'
      % np.abs(d.dev.down(p.w, (MM.MD, MM.WROW)).astype(np.float32)).max())
