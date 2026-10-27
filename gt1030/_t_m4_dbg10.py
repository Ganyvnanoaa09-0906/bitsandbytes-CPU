# -*- coding: utf-8 -*-
"""M4 debug 10: 往三个不同权重 buffer 写不同常数，再回读，看是否互相串。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
model = MM.V2ModelDev(ckpt)
d = model.dev
q = d.q
targets = [('L0.Wqkv', model.layers[0].W['Wqkv'].w, (MM.MD, MM.WROW), 1.5),
           ('L0.W3', model.layers[0].W['W3'].w, (MM.FF, MM.MD), 2.5),
           ('head', model.head.w, (MM.MD, model.V), 3.5)]
for nm, b, sh, val in targets:
    cl.enqueue_copy(q, b, np.full(int(np.prod(sh)), val, np.float16))
q.finish()
for nm, b, sh, val in targets:
    a = d.down(b, sh).astype(np.float32)
    print('%-10s expect %.1f  got absmax=%.3f mean=%.4f' % (nm, val, np.abs(a).max(), a.mean()))
# 再读回原始（未覆盖前）不该出现
print('--- 现在把 L0.Wqkv 读成完整数组，和 L0.W3 比 ---')
a1 = d.down(targets[0][1], (MM.MD, MM.WROW)).astype(np.float32)
a2 = d.down(targets[1][1], (MM.FF, MM.MD)).astype(np.float32)
print('a1 mean=%.4f a2 mean=%.4f  same? %s' % (a1.mean(), a2.mean(), np.allclose(a1, a2)))
