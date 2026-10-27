# -*- coding: utf-8 -*-
"""探针 9：按真实尺寸逐个 up()，每次都回读校验，找出第一次读到脏数据的位置。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM
import gt1030.attention_gpu as A

ctx, q, mf = A.ctx, A.queue, A.mf
MB = 1 << 20
n_w = {'Wqkv': MM.MD * MM.WROW, 'Wo': MM.MD * MM.MD,
       'W12': MM.MD * 2 * MM.FF, 'W3': MM.FF * MM.MD}
lsz = []
for k, n in n_w.items():
    lsz += [n * 2, n * 2, n * 4, n * 4, n * 4]
lsz += [MM.MD * 2] * 2 + [MM.MD * 4] * 4
act = {'xin': MM.M * MM.MD, 'h1': MM.M * MM.MD, 'h2': MM.M * MM.MD, 'x1': MM.M * MM.MD,
       'o_tok': MM.M * MM.MD, 'P': MM.BH * MM.T * MM.T, 'Qn': MM.BH * MM.T * MM.HD,
       'Kn': MM.BH * MM.T * MM.HD, 'Qraw': MM.BH * MM.T * MM.HD, 'Kraw': MM.BH * MM.T * MM.HD,
       'Vb': MM.BH * MM.T * MM.HD, 'a2': MM.M * 2 * MM.FF, 'g': MM.M * MM.FF}
lsz += [n * 2 for n in act.values()]
plan = [('L%d' % i, nb) for i in range(10) for nb in lsz]

bufs = []
total = 0
for i, (tag, nb) in enumerate(plan):
    val = float(i + 1)          # fp16 能精确表示的小整数
    a = np.full(nb // 2, val, np.float16)
    b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
    bufs.append(b)
    total += nb
    out = np.empty(min(nb // 2, 4096), np.float16)
    cl.enqueue_copy(q, out, b)
    q.finish()
    got = float(out[0])
    if got != val:
        print('*** 第 %d 个 (%s, %.1fMB) 读回 %.3f 期望 %.3f  累计 %.2f GB'
              % (i, tag, nb / 1e6, got, val, total / 1e9))
        # 再看看最早那块
        o2 = np.empty(1024, np.float16)
        cl.enqueue_copy(q, o2, bufs[0])
        q.finish()
        print('    第一块读回 = %.3f (期望 1.000)' % float(o2[0]))
        break
    if i % 50 == 0:
        print('  #%3d %-4s %.1fMB  累计 %.2f GB  ok' % (i, tag, nb / 1e6, total / 1e9))
else:
    print('全部 %d 个 buffer 读写正常，累计 %.2f GB' % (len(bufs), total / 1e9))
