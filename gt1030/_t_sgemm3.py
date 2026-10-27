# -*- coding: utf-8 -*-
"""探针 46：自写 s_gemm 到底写了什么？直接和 numpy 对拍一个 tile。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM
import gt1030.attention_gpu as A

T, HD, BH = 1024, 64, 16
KSRC = open(r'D:\work\bitsandbytes-CPU\_sgemm_src.cl', 'r', encoding='utf-8').read() \
    if False else r"""
__kernel void s_gemm(__global const half* Q, __global const half* Kt,
                     __global half* S, float scale) {
    int bh = get_global_id(2), ti = get_global_id(0), tj = get_global_id(1);
    __local half Qs[64][64];
    __local half Ks[64][64];
    int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int lx = get_local_id(0), ly = get_local_id(1);
    __global const half* Qp = Q + (long)bh * 1024 * 64;
    __global const half* Kp = Kt + (long)bh * 1024 * 64;
    __global half* Sp = S + (long)bh * 1024 * 1024;
    for (int i = lid; i < 4096; i += 256) {
        int r = i / 64, c = i % 64;
        Qs[r][c] = Qp[(long)(ti * 64 + r) * 64 + c];
        Ks[r][c] = Kp[(long)(tj * 64 + r) * 64 + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int r0 = ly * 4, c0 = lx * 4;
    float acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int kk = 0; kk < 64; ++kk) {
        float qv[4], kv[4];
        for (int i = 0; i < 4; ++i) qv[i] = (float)Qs[r0 + i][kk];
        for (int j = 0; j < 4; ++j) kv[j] = (float)Ks[c0 + j][kk];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) acc[i][j] += qv[i] * kv[j];
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Sp[(long)(ti * 64 + r0 + i) * 1024 + (tj * 64 + c0 + j)] = (half)(acc[i][j] * scale);
}
"""
ctx, q_ = A.ctx, A.queue
prog = cl.Program(ctx, KSRC).build(options=['-cl-mad-enable'])
kern = cl.Kernel(prog, 's_gemm')
mf = cl.mem_flags
rng = np.random.default_rng(0)
Qh = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
Kh = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
Qb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Qh)
Kb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Kh)
S2 = cl.Buffer(ctx, mf.READ_WRITE, BH * T * T * 2)
# 先清零，确认 kernel 真的写了
z = np.zeros(BH * T * T, np.float16)
cl.enqueue_copy(q_, S2, z, is_blocking=True)
q_.finish()
SC = 1.0 / np.sqrt(HD)
kern(q_, (T // 64, T // 64, BH), (16, 16, 1), Qb, Kb, S2, np.float32(SC))
q_.finish()
out = np.empty(BH * T * T, np.float16)
cl.enqueue_copy(q_, out, S2)
q_.finish()
out = out.reshape(BH, T, T).astype(np.float32)
ref = (Qh.astype(np.float32) @ Kh.astype(np.float32).transpose(0, 2, 1)) * SC
print('全零? %s  非零个数 %d / %d' % (bool((out == 0).all()), int((out != 0).sum()), out.size))
print('max|out|=%.4f  max|ref|=%.4f' % (np.abs(out).max(), np.abs(ref).max()))
print('tile(0,0): out[0,0,:4]=%s ref[0,0,:4]=%s'
      % (np.round(out[0, 0, :4], 3), np.round(ref[0, 0, :4], 3)))
print('tile(1,2): out[0,64,128:132]=%s ref=%s'
      % (np.round(out[0, 64, 128:132], 3), np.round(ref[0, 64, 128:132], 3)))
rel = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-12))
print('rel=%.3e' % rel)
