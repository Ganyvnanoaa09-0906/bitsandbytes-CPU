# -*- coding: utf-8 -*-
"""M4 debug 9: 检查 cl_mem 句柄是否重复（驱动复用/句柄回绕）。"""
import sys
from collections import Counter
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import gt1030.ar_v2_model as MM

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
model = MM.V2ModelDev(ckpt)
handles = {}
dups = []
for i, ly in enumerate(model.layers):
    for k, p in ly.W.items():
        for nm, b in (('w', p.w), ('grad', p.grad), ('master', p.master)):
            h = int(b.int_ptr)
            if h in handles:
                dups.append((i, k, nm, h, handles[h]))
            handles[h] = (i, k, nm)
    for k, b in ly.b.items():
        h = int(b.int_ptr)
        if h in handles:
            dups.append((i, k, h, handles[h]))
        handles[h] = (i, k)
print('total distinct-looking buffers: %d' % len(handles))
print('duplicate handles: %d' % len(dups))
for d in dups[:10]:
    print('  ', d)
print('layer0 Wqkv ptr=%#x   layer0 W3 ptr=%#x   head ptr=%#x'
      % (model.layers[0].W['Wqkv'].w.int_ptr, model.layers[0].W['W3'].w.int_ptr,
         model.head.w.int_ptr))
