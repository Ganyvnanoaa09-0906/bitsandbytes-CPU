# -*- coding: utf-8 -*-
"""小测试：直接调 Dev.up 看校验为什么失败。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

d = MM.Dev()
a = (np.random.default_rng(0).standard_normal((512, 1024)) * 0.03).astype(np.float16)
print('host[:4]', a.ravel()[:4])
try:
    b = d.up(a)
    print('up OK')
except MM.DriverBroken as e:
    print('up raised', e)
    # 手工重现
    bb = cl.Buffer(d.ctx, d.mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(d.q, bb, np.ascontiguousarray(a), is_blocking=True)
    d.q.finish()
    chk = np.empty(4, np.float16)
    cl.enqueue_copy(d.q, chk, bb)
    d.q.finish()
    print('manual read[:4]', chk)
    big = np.empty(a.size, np.float16)
    cl.enqueue_copy(d.q, big, bb)
    d.q.finish()
    print('manual read full[:4]', big[:4], ' absmax=%.4f' % np.abs(big).max())
