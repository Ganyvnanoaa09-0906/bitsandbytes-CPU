# -*- coding: utf-8 -*-
"""探针 29：找到本 context 的 buffer 数量上限（逐步加 4KB dummy，每次校验 sentinel）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

CKPT = r'F:\weibo_cos\ckpt_1l.pt'
dev = MM.V2ModelDev(CKPT)
d = dev.dev
q, ctx, mf = d.q, d.ctx, d.mf

# 先数一下模型用了多少 buffer
n_model = 0
for ly in dev.layers:
    n_model += len(ly.b) + 4 * 5
n_model += len(dev.sc) + 8
print('模型约 %d 个 buffer' % n_model)

sent = cl.Buffer(ctx, mf.READ_WRITE, 4096)
pat = np.full(2048, 0.75, np.float16)
cl.enqueue_copy(q, sent, pat, is_blocking=True)
q.finish()
dummies = []
bad = None
for i in range(400):
    dummies.append(cl.Buffer(ctx, mf.READ_WRITE, 4096))
    # 每次写 0.25+i 到新块，再校验 sentinel
    v = np.full(2048, 0.25 + i * 0.5, np.float16)
    cl.enqueue_copy(q, dummies[-1], v, is_blocking=True)
    out = np.empty(2048, np.float16)
    cl.enqueue_copy(q, out, sent)
    q.finish()
    o2 = np.empty(2048, np.float16)
    cl.enqueue_copy(q, o2, dummies[-1])
    q.finish()
    if (not np.array_equal(out, pat) or not np.array_equal(o2, v)) and bad is None:
        bad = i
        print('*** 第 %d 个额外 buffer 后失效（总 buffer ≈ %d）: sentinel_ok=%s newest_ok=%s'
              % (i, n_model + i + 2, np.array_equal(out, pat), np.array_equal(o2, v)))
        break
print('加了 %d 个 dummy, bad=%s' % (len(dummies), bad))
