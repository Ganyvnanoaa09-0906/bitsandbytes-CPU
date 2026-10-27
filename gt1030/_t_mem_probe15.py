# -*- coding: utf-8 -*-
"""探针 15：分阶段构建整机，每阶段后做一次『新 buffer 复制→回读』自检，定位队列何时死。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

d = MM.Dev()
q = d.q


def alive(tag):
    try:
        a = np.full(512, 3.0, np.float16)
        b = cl.Buffer(d.ctx, d.mf.READ_WRITE, 1024)
        cl.enqueue_copy(q, b, a, is_blocking=True)
        out = np.empty(512, np.float16)
        cl.enqueue_copy(q, out, b)
        q.finish()
        v = float(out[0])
        print('%-22s queue_alive=%-5s value=%.2f' % (tag, v == 3.0, v))
        return v == 3.0
    except Exception as e:
        print('%-22s EXC %r' % (tag, e))
        return False


alive('after Dev()')
ck = __import__('torch').load(r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt',
                             map_location='cpu', weights_only=False)
sd = ck['model']
keep = []
for i in range(10):
    keep.append(MM.Layer(d, sd, i))
    if not alive('after Layer %d' % i):
        break
