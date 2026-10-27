# -*- coding: utf-8 -*-
"""M4 debug 7: 用可识别图案 + enqueue_fill 检验 buffer 是否被复用/清零。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

dev = MM.Dev()
q, mf = dev.q, dev.mf
N = 2048


def mk(val):
    a = np.full(N, val, np.float16)
    return dev.up(a), a


def rd(b):
    return float(dev.down(b, (N,))[0])


bufs = [mk(float(i + 1))[0] for i in range(5)]
print('初始:', [rd(b) for b in bufs])

# 大量小 buffer + 一些大 buffer，模拟整机
small = [dev.buf(4096) for _ in range(400)]
big = [dev.buf(32 << 20) for _ in range(20)]
print('分配 420 个 buffer 后:', [rd(b) for b in bufs])

# 用 enqueue_fill 覆盖一批 small，再读回 bufs
for b in small[:100]:
    cl.enqueue_fill(q, b, np.float32(0).tobytes(), 0, 4096)
q.finish()
print('fill 100 个小 buffer 后:', [rd(b) for b in bufs])
print('small[0] 内容:', float(dev.down(small[0], (2048,))[0]) if small else None)
