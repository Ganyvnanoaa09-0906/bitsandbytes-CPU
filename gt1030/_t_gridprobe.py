# -*- coding: utf-8 -*-
"""探针 47：确定 s_gemm 实际写了哪些 tile（把 bh/ti/tj 打出来）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM
import gt1030.attention_gpu as A

ctx, q_ = A.ctx, A.queue
KSRC = r"""
__kernel void probe_ids(__global int* out) {
    int g = get_global_id(2) * 256 + get_global_id(1) * 16 + get_global_id(0);
    out[g * 3 + 0] = get_global_id(0);
    out[g * 3 + 1] = get_global_id(1);
    out[g * 3 + 2] = get_global_id(2);
}
"""
prog = cl.Program(ctx, KSRC).build()
kern = cl.Kernel(prog, 'probe_ids')
mf = cl.mem_flags
out = cl.Buffer(ctx, mf.READ_WRITE, 16 * 16 * 16 * 3 * 4)
kern(q_, (16, 16, 16), (16, 16, 1), out)
q_.finish()
a = np.empty((16 * 16 * 16, 3), np.int32)
cl.enqueue_copy(q_, a.ravel(), out)
q_.finish()
print('前 6 个组 (gid0,gid1,gid2):')
print(a[:6])
print('gid0 取值集合:', np.unique(a[:, 0]))
print('gid1 取值集合:', np.unique(a[:, 1]))
print('gid2 取值集合:', np.unique(a[:, 2]))
