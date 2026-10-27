# -*- coding: utf-8 -*-
"""探针 8：把 Dev.up 改成『先分配再显式 copy』（不用 COPY_HOST_PTR），看读写是否恢复。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM


def up2(self, a):
    a = np.ascontiguousarray(a)
    b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(self.q, b, a)
    return b


MM.Dev.up = up2

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
d = MM.Dev()
q = d.q
sent = d.up(np.full(4096, 0.5, np.float16))
print('sentinel before build: %.3f' % float(d.down(sent, (4096,))[0]))
model = MM.V2ModelDev(ckpt, dev=d)
print('sentinel after build : %.3f' % float(d.down(sent, (4096,))[0]))
p = model.layers[0].W['Wqkv']
a = d.down(p.w, (MM.MD, MM.WROW)).astype(np.float32)
print('L0.Wqkv absmax=%.4f mean=%.5f' % (np.abs(a).max(), a.mean()))
nb = d.up(np.full(8192, 0.25, np.float16))
print('new buffer           : %.3f' % float(d.down(nb, (8192,))[0]))
tokens = np.zeros((MM.B, MM.T), np.int64)
model.forward(tokens)
lg = d.down(model.sc['logits'], (MM.M, model.V), np.float16).astype(np.float32)
print('logits absmax=%.4f' % np.abs(lg).max())
