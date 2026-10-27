# -*- coding: utf-8 -*-
"""探针：iGPU 上持续分配 buffer，检查最早写入的内容何时读回变成 0（WDDM 逐出）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import psutil
import gt1030.attention_gpu as A

ctx, q, mf = A.ctx, A.queue, A.mf
dev0 = ctx.devices[0]
print('device global_mem=%.2fGB max_alloc=%.2fGB'
      % (dev0.global_mem_size / 1e9, dev0.max_mem_alloc_size / 1e9))

MB = 1 << 20
step = 64 * MB
blocks = []
first = None
N = 24
for i in range(N):
    a = np.full(step // 2, 0.5, np.float16)          # 64MB 的 half
    b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
    blocks.append(b)
    if first is None:
        first = b
    # 每次都读回【第一块】，看还在不在
    out = np.empty(step // 2, np.float16)
    cl.enqueue_copy(q, out, first)
    q.finish()
    v = float(out[0])
    vm = psutil.virtual_memory()
    print('alloc %3d MB total=%5d MB  first_block[0]=%.3f  host_avail=%.1fGB  %s'
          % ((i + 1) * (step // MB), (i + 1) * (step // MB), v, vm.available / 1e9,
             '' if v == 0.5 else '  <-- CORRUPTED'))
