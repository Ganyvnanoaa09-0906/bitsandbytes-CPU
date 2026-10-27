# -*- coding: utf-8 -*-
"""探针 13：同一进程里，构建前后各做一次『复制→回读』，并在构建后重传 ln1。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

d = MM.Dev()
q = d.q


def rt(tag, val=1.0):
    a = np.full(1024, val, np.float16)
    b = cl.Buffer(d.ctx, d.mf.READ_WRITE, 2048)
    cl.enqueue_copy(q, b, a, is_blocking=True)
    out = np.empty(1024, np.float16)
    cl.enqueue_copy(q, out, b)
    q.finish()
    print('%-24s -> %.3f (期望 %.1f)' % (tag, float(out[0]), val))
    return b


rt('before build')
ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
model = MM.V2ModelDev(ckpt, dev=d)
rt('after build', 2.0)

ln1 = model.layers[0].ln1
print('ln1 read          -> %.4f' % np.abs(d.down(ln1, (MM.MD,), np.float16).astype(np.float32)).max())
# 重新上传 ln1
w = np.full(MM.MD, 1.05, np.float16)
cl.enqueue_copy(q, ln1, w, is_blocking=True)
q.finish()
print('ln1 after rewrite -> %.4f' % np.abs(d.down(ln1, (MM.MD,), np.float16).astype(np.float32)).max())
xa = model.xa
xh = np.full(MM.M * MM.MD, 0.5, np.float16)
cl.enqueue_copy(q, xa, xh, is_blocking=True)
q.finish()
print('xa after write    -> %.4f' % np.abs(d.down(xa, (MM.M, MM.MD), np.float16).astype(np.float32)).max())
