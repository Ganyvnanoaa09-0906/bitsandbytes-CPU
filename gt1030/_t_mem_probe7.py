# -*- coding: utf-8 -*-
"""探针 7：完整复刻 ar_v2_model 的分配序列（含 scratch），找断点。"""
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
layer_sizes = []
for k, n in n_w.items():
    layer_sizes += [n * 2, n * 2, n * 4, n * 4, n * 4]
layer_sizes += [MM.MD * 2] * 2
layer_sizes += [MM.MD * 4] * 4
act = {'xin': MM.M * MM.MD, 'h1': MM.M * MM.MD, 'h2': MM.M * MM.MD, 'x1': MM.M * MM.MD,
       'o_tok': MM.M * MM.MD, 'P': MM.BH * MM.T * MM.T, 'Qn': MM.BH * MM.T * MM.HD,
       'Kn': MM.BH * MM.T * MM.HD, 'Qraw': MM.BH * MM.T * MM.HD, 'Kraw': MM.BH * MM.T * MM.HD,
       'Vb': MM.BH * MM.T * MM.HD, 'a2': MM.M * 2 * MM.FF, 'g': MM.M * MM.FF}
layer_sizes += [n * 2 for n in act.values()]
scratch = {'qkv': MM.M * MM.WROW, 'S': MM.BH * MM.T * MM.T, 'O': MM.BH * MM.T * MM.HD,
           'attn': MM.M * MM.MD, 'y': MM.M * MM.MD, 'logits': MM.M * 1024,
           'dqkv': MM.M * MM.WROW, 'dy': MM.M * MM.MD, 'dg': MM.M * MM.FF,
           'da2': MM.M * 2 * MM.FF, 'dh2': MM.M * MM.MD, 'dx1mlp': MM.M * MM.MD,
           'dattn': MM.M * MM.MD, 'dO_tok': MM.M * MM.MD, 'dO': MM.BH * MM.T * MM.HD,
           'dP': MM.BH * MM.T * MM.T, 'dS': MM.BH * MM.T * MM.T, 'dQ': MM.BH * MM.T * MM.HD,
           'dK': MM.BH * MM.T * MM.HD, 'dV': MM.BH * MM.T * MM.HD,
           'dQraw': MM.BH * MM.T * MM.HD, 'dKraw': MM.BH * MM.T * MM.HD,
           'dh1': MM.M * MM.MD, 'dxattn': MM.M * MM.MD, 'dtmp': MM.M * MM.MD,
           'dlogits': MM.M * 1024}
print('计划: 10 层 × %.1f MB = %.2f GB + scratch %.2f GB'
      % (sum(layer_sizes) / 1e6, sum(layer_sizes) * 10 / 1e9,
         sum(scratch.values()) * 2 / 1e9))

sent = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                 hostbuf=np.full(1024, 0.5, np.float16))
bufs = [sent]
total = 0
bad = None
plan = [('L%d' % i, nb) for i in range(10) for nb in layer_sizes]
plan += [('S.' + k, nb * 2) for k, nb in scratch.items()]
for i, (tag, nb) in enumerate(plan):
    b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                  hostbuf=np.zeros(nb // 2, np.float16))
    bufs.append(b)
    total += nb
    out = np.empty(1024, np.float16)
    cl.enqueue_copy(q, out, sent)
    q.finish()
    ok = float(out[0]) == 0.5
    if not ok and bad is None:
        bad = i
        print('*** 断点：第 %d 个分配 (%s, +%.1fMB) 之后 sentinel 变 %.3f；累计 %.2f GB'
              % (i, tag, nb / 1e6, float(out[0]), total / 1e9))
        break
    if i % 40 == 0:
        print('  #%3d %-6s 累计 %.2f GB  sentinel=%.2f' % (i, tag, total / 1e9, float(out[0])))
print('结束：%d 个 buffer, 累计 %.2f GB, bad=%s' % (len(bufs), total / 1e9, bad))
