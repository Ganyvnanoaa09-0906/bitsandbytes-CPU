# -*- coding: utf-8 -*-
"""探针：按整机真实的 buffer 数量/尺寸分配，找出从第几个开始读写失效。"""
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
sizes = []
for k, n in n_w.items():
    sizes += [n * 2, n * 2, n * 4, n * 4, n * 4]        # w, grad, master, mom, vel
sizes += [MM.MD * 2] * 2                                # ln1/ln2 (half)
sizes += [MM.MD * 4] * 4                                 # qw, kw, ls1, ls2 (float)
act = {'xin': MM.M * MM.MD, 'h1': MM.M * MM.MD, 'h2': MM.M * MM.MD, 'x1': MM.M * MM.MD,
       'o_tok': MM.M * MM.MD, 'P': MM.BH * MM.T * MM.T, 'Qn': MM.BH * MM.T * MM.HD,
       'Kn': MM.BH * MM.T * MM.HD, 'Qraw': MM.BH * MM.T * MM.HD, 'Kraw': MM.BH * MM.T * MM.HD,
       'Vb': MM.BH * MM.T * MM.HD, 'a2': MM.M * 2 * MM.FF, 'g': MM.M * MM.FF}
sizes += [n * 2 for n in act.values()]
per_layer = sum(sizes)
print('每层 buffer 数 = %d, 每层 %.1f MB' % (len(sizes), per_layer / 1e6))

bufs = []
first = None
bad = None
for i, nb in enumerate(sizes * 12):
    a = np.full(nb // 2, 0.5, np.float16)
    b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
    bufs.append(b)
    if first is None:
        first = b
    out = np.empty(min(nb, 2 * MB) // 2, np.float16)
    cl.enqueue_copy(q, out, first)
    q.finish()
    v = float(out[0])
    if i > 0:
        cl.enqueue_copy(q, b, np.full(nb // 2, 1.5, np.float16))
    o2 = np.empty(8, np.float16)
    cl.enqueue_copy(q, o2, b)
    q.finish()
    v2 = float(o2[0])
    if (v != 0.5 or v2 != 1.5) and bad is None:
        bad = i
        tot = (i // len(sizes)) * per_layer + sum(sizes[:i % len(sizes) + 1])
        print('*** 第 %d 个 buffer 起失效（累计 %.1f MB，第 %d 层）: first=%.2f newest=%.2f'
              % (i, tot / 1e6, i // len(sizes), v, v2))
        break
tot = sum(bu.size for bu in bufs)
print('分配了 %d 个 buffer, 累计 %.2f GB' % (len(bufs), tot / 1e9))
if bad is None:
    print('全部正常（没有失效点）')
