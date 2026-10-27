# -*- coding: utf-8 -*-
"""探针 6：构建后，检查『读路径』是否还有效：旧 buffer / 新建 buffer / 重写后读。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
d = MM.Dev()
q = d.q
sent = d.up(np.full(4096, 0.5, np.float16))
model = MM.V2ModelDev(ckpt, dev=d)
print('1) sentinel   ->  %.3f' % float(d.down(sent, (4096,))[0]))

p = model.layers[0].W['Wqkv']
print('2) L0.Wqkv    ->  absmax %.4f'
      % np.abs(d.down(p.w, (MM.MD, MM.WROW)).astype(np.float32)).max())

# 新建一块，写 0.25 再读
nb = d.up(np.full(8192, 0.25, np.float16))
print('3) new buffer ->  %.3f' % float(d.down(nb, (8192,))[0]))

# 重写 L0.Wqkv 为 1.5 再读
cl.enqueue_copy(q, p.w, np.full(p.n, 1.5, np.float16))
q.finish()
print('4) L0.Wqkv rewritten 1.5 ->  absmax %.4f / mean %.4f'
      % (np.abs(d.down(p.w, (MM.MD, MM.WROW)).astype(np.float32)).max(),
         float(d.down(p.w, (MM.MD, MM.WROW)).astype(np.float32).mean())))
# 再读 sentinel
print('5) sentinel   ->  %.3f' % float(d.down(sent, (4096,))[0]))
