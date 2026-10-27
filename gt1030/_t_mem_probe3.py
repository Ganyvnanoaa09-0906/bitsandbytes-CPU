# -*- coding: utf-8 -*-
"""探针 3：只验证『最早那块 sentinel 的内容会不会失效』，按整机真实尺寸分配。"""
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
    sizes += [n * 2, n * 2, n * 4, n * 4, n * 4]
sizes += [MM.MD * 2] * 2
sizes += [MM.MD * 4] * 4
act = {'xin': MM.M * MM.MD, 'h1': MM.M * MM.MD, 'h2': MM.M * MM.MD, 'x1': MM.M * MM.MD,
       'o_tok': MM.M * MM.MD, 'P': MM.BH * MM.T * MM.T, 'Qn': MM.BH * MM.T * MM.HD,
       'Kn': MM.BH * MM.T * MM.HD, 'Qraw': MM.BH * MM.T * MM.HD, 'Kraw': MM.BH * MM.T * MM.HD,
       'Vb': MM.BH * MM.T * MM.HD, 'a2': MM.M * 2 * MM.FF, 'g': MM.M * MM.FF}
sizes += [n * 2 for n in act.values()]
per_layer = sum(sizes)
print('每层 %d buffer / %.1f MB；计划 10 层 = %.2f GB'
      % (len(sizes), per_layer / 1e6, per_layer * 10 / 1e9))

sent = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                 hostbuf=np.full(1024, 0.5, np.float16))
bufs = [sent]
bad = None
total = 0
for i, nb in enumerate(sizes * 10):
    b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                  hostbuf=np.zeros(nb // 2, np.float16))
    bufs.append(b)
    total += nb
    out = np.empty(1024, np.float16)
    cl.enqueue_copy(q, out, sent)
    q.finish()
    if float(out[0]) != 0.5 and bad is None:
        bad = i
        print('*** sentinel 在第 %d 次分配后失效（累计 %.2f GB, 第 %d 层）: %.3f'
              % (i, total / 1e9, i // len(sizes), float(out[0])))
        break
print('分配 %d 个 buffer, 累计 %.2f GB' % (len(bufs), total / 1e9))
if bad is None:
    print('sentinel 全部正常 —— 说明 10 层规模本身没问题')
