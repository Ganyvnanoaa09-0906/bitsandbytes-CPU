# -*- coding: utf-8 -*-
"""探针 26：device->device 拷贝用 byte_count= 到底有没有生效？"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import inspect
import gt1030.ar_v2_model as MM

d = MM.Dev()
q, ctx, mf = d.q, d.ctx, d.mf
n = 1 << 16
src = cl.Buffer(ctx, mf.READ_WRITE, n * 2)
dst = cl.Buffer(ctx, mf.READ_WRITE, n * 2)
v = np.arange(n, dtype=np.float16)
cl.enqueue_copy(q, src, v, is_blocking=True)
q.finish()
# 1) 带 byte_count=
cl.enqueue_copy(q, dst, src, byte_count=n * 2)
q.finish()
out = np.empty(n, np.float16)
cl.enqueue_copy(q, out, dst)
q.finish()
print('with byte_count= : equal=%s out[:4]=%s' % (np.array_equal(out, v), out[:4]))
# 2) 不带 kwarg
cl.enqueue_copy(q, dst, src)
q.finish()
cl.enqueue_copy(q, out, dst)
q.finish()
print('no kwarg         : equal=%s out[:4]=%s' % (np.array_equal(out, v), out[:4]))
# 3) 用 cl.enqueue_copy 的正确签名试试
try:
    cl.enqueue_copy(q, dst, src, src_offset=0, dst_offset=0, size=n * 2)
    q.finish()
    cl.enqueue_copy(q, out, dst)
    q.finish()
    print('with size=       : equal=%s out[:4]=%s' % (np.array_equal(out, v), out[:4]))
except Exception as e:
    print('size= 不支持: %r' % (e,))
