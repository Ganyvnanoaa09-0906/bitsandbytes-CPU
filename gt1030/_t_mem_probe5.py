# -*- coding: utf-8 -*-
"""探针 5：same-cl_mem? 打印 sentinel 与权重 buffer 的 int_ptr 和 size。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import gt1030.ar_v2_model as MM

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
d = MM.Dev()
sent = d.up(np.full(4096, 0.5, np.float16))
print('sentinel ptr=%#x size=%d' % (sent.int_ptr, sent.size))
model = MM.V2ModelDev(ckpt, dev=d)
for i in (0, 5, 9):
    for k in ('Wqkv', 'W3'):
        p = model.layers[i].W[k]
        print('L%d %-4s ptr=%#x size=%d  n=%d' % (i, k, p.w.int_ptr, p.w.size, p.n))
print('head ptr=%#x size=%d' % (model.head.w.int_ptr, model.head.w.size))
print('sentinel again ptr=%#x size=%d' % (sent.int_ptr, sent.size))
