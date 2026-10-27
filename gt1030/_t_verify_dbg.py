# -*- coding: utf-8 -*-
"""直接测：ln1(512 个 fp16) / master(524288 个 fp32) 的上传+4KB 回读校验。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

d = MM.Dev()
q, ctx, mf = d.q, d.ctx, d.mf
rng = np.random.default_rng(0)
cases = [('ln1', np.ones(512, np.float16)), ('qw', np.ones(64, np.float32)),
         ('master', (rng.standard_normal(524288) * 0.03).astype(np.float32)),
         ('Wqkv_raw', (rng.standard_normal((512, 1024)) * 0.03).astype(np.float16))]
for name, a in cases:
    a = np.ascontiguousarray(a)
    b = cl.Buffer(ctx, mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(q, b, a, is_blocking=True)
    q.finish()
    nb = min(a.nbytes, 4096)
    nb -= nb % 8
    chk = np.empty(nb, np.uint8)
    cl.enqueue_copy(q, chk, b)
    q.finish()
    want = a.ravel().view(np.uint8)[:nb]
    eq = np.array_equal(chk, want)
    print('%-10s nbytes=%7d nb=%4d equal=%-5s  chk[:8]=%s want[:8]=%s'
          % (name, a.nbytes, nb, eq, list(chk[:8]), list(want[:8])))
    if not eq:
        full = np.empty(a.size, a.dtype)
        cl.enqueue_copy(q, full, b)
        q.finish()
        print('     full_equal=%s' % np.array_equal(full, a.ravel()))
