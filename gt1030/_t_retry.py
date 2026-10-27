# -*- coding: utf-8 -*-
"""探针 17：同一进程内『重建 context → 构建整机 → 自检』重试 5 次，看重试能否救回。"""
import sys
import ctypes
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.attention_gpu as A
import gt1030.ar_v2_model as MM

CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'


def fresh_context():
    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx)
    A.ctx, A.queue, A.mf = ctx, q, cl.mem_flags
    A.QP = ctypes.c_void_p(int(q.int_ptr))
    return ctx, q


for attempt in range(1, 6):
    ctx, q = fresh_context()
    d = MM.Dev()
    model = MM.V2ModelDev(CKPT, dev=d)
    a = d.down(model.layers[0].W['Wqkv'].w, (MM.MD, MM.WROW)).astype(np.float32)
    good = abs(float(np.abs(a).max()) - 0.1143) < 5e-3
    # 再来一次自检：新 buffer 复制→回读
    ab = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 1024)
    cl.enqueue_copy(q, ab, np.full(512, 7.0, np.float16), is_blocking=True)
    out = np.empty(512, np.float16)
    cl.enqueue_copy(q, out, ab)
    q.finish()
    alive = float(out[0]) == 7.0
    print('attempt %d: weights_ok=%s (absmax=%.4f)  queue_alive=%s' % (attempt, good, float(np.abs(a).max()), alive))
    if good and alive:
        print('  -> 重试成功')
        break
    del model, d
    ctx = None
