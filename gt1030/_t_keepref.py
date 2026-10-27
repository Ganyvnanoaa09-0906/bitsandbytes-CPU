# -*- coding: utf-8 -*-
"""测试假说：驱动对大 buffer 的『blocking 写』其实不是真同步，函数返回后源 host 数组就被
GC 回收/复用 -> 上传的数据变成垃圾。对策：上传时自己保留一份 host 副本的引用。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

d = MM.Dev()
q, ctx, mf = d.q, d.ctx, d.mf
rng = np.random.default_rng(0)
KEEP = []


def upload(a, keep=False):
    a = np.ascontiguousarray(a)
    if keep:
        a = np.array(a, copy=True)
        KEEP.append(a)
    b = cl.Buffer(ctx, mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(q, b, a, is_blocking=True)
    q.finish()
    back = np.empty(a.size, a.dtype)
    cl.enqueue_copy(q, back, b)
    q.finish()
    return np.array_equal(back, a.ravel())


bad_nokeep = bad_keep = 0
for trial in range(30):
    shp = (512, 1024)
    src = (rng.standard_normal(shp) * 0.03).astype(np.float16)
    f32 = src.astype(np.float32).ravel()          # 临时数组（模拟 Param.master）
    if not upload(f32, keep=False):
        bad_nokeep += 1
    f32b = src.astype(np.float32).ravel()
    if not upload(f32b, keep=True):
        bad_keep += 1
print('不保留引用: %d/30 失败   保留引用: %d/30 失败' % (bad_nokeep, bad_keep))
