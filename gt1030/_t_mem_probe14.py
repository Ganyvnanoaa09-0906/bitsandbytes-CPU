# -*- coding: utf-8 -*-
"""探针 14：分别用 2MB / 32MB buffer 做『分配+写入+回读』，找出失效点（数量 vs 总量）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

MB = 1 << 20


def probe(buf_mb, max_n):
    d = MM.Dev()
    q, ctx, mf = d.q, d.ctx, d.mf
    nb = buf_mb * MB
    n_half = nb // 2
    bufs = []
    for i in range(max_n):
        a = np.full(n_half, float(i + 1), np.float16)
        b = cl.Buffer(ctx, mf.READ_WRITE, nb)
        cl.enqueue_copy(q, b, a, is_blocking=True)
        out = np.empty(n_half, np.float16)
        cl.enqueue_copy(q, out, b)
        q.finish()
        got = float(out[0])
        if got != float(i + 1):
            print('  %dMB x %d: 第 %d 个失效（读回 %.1f 期望 %d，累计 %.2f GB）'
                  % (buf_mb, max_n, i, got, i + 1, (i + 1) * buf_mb / 1024.0))
            return i, (i + 1) * buf_mb / 1024.0
        bufs.append(b)
    print('  %dMB x %d: 全部正常（累计 %.2f GB）' % (buf_mb, max_n, max_n * buf_mb / 1024.0))
    return max_n, max_n * buf_mb / 1024.0


print('== 2MB buffer ==')
probe(2, 700)
print('== 32MB buffer ==')
probe(32, 60)
print('== 8MB buffer ==')
probe(8, 250)
