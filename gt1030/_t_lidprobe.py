# -*- coding: utf-8 -*-
"""探针 49：最小化定位 —— (a) local id 是否正常；(b) __local 数组能不能正确搬运。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM
import gt1030.attention_gpu as A

ctx, q_ = A.ctx, A.queue
SRC = r"""
__kernel void lidprobe(__global int* out) {
    int lid = get_local_id(1) * 16 + get_local_id(0);
    out[get_global_id(1) * 16 + get_global_id(0)] = lid * 1000 + get_global_id(0) * 100
                                                   + get_local_id(1) * 10 + get_local_id(0);
}
__kernel void localprobe(__global const float* src, __global float* dst) {
    __local float buf[64][64];
    int lid = get_local_id(1) * 16 + get_local_id(0);
    int lx = get_local_id(0), ly = get_local_id(1);
    for (int i = lid; i < 4096; i += 256) {
        int r = i / 64, c = i % 64;
        buf[r][c] = src[(long)(get_global_id(0) * 64 + r) * 64 + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int r = ly * 4 + i, c = lx * 4 + j;
            dst[(long)(get_global_id(0) * 64 + r) * 64 + c] = buf[r][c];
        }
}
"""
prog = cl.Program(ctx, SRC).build()
mf = cl.mem_flags
o = cl.Buffer(ctx, mf.READ_WRITE, 16 * 16 * 4)
cl.Kernel(prog, 'lidprobe')(q_, (16, 16), (16, 16), o)
q_.finish()
a = np.empty((16, 16), np.int32); cl.enqueue_copy(q_, a.ravel(), o); q_.finish()
print('lidprobe 前 3x3:\n', a[:3, :3])
print('lidprobe 每行不同? %s  unique=%d' % (a[0, 0] != a[1, 0], len(np.unique(a))))

src = (np.arange(2 * 64 * 64, dtype=np.float32) * 1.0).reshape(2 * 64 * 64)
sb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src)
db = cl.Buffer(ctx, mf.READ_WRITE, 2 * 64 * 64 * 4)
cl.Kernel(prog, 'localprobe')(q_, (2,), (16, 16), sb, db)
q_.finish()
out = np.empty(2 * 64 * 64, np.float32); cl.enqueue_copy(q_, out, db); q_.finish()
print('localprobe 一致: %s  (first=%s)' % (np.array_equal(out, src), out[:4]))
