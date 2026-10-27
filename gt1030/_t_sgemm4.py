# -*- coding: utf-8 -*-
"""探针 48：s_gemm 只跑一个工作组（1 个 tile），和 numpy 对拍，验证 kernel 逻辑本身。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM
import gt1030.attention_gpu as A

ctx, q_ = A.ctx, A.queue
KSRC = r"""
// 简化：一个组算 64x64 的 tile，(ti,tj,bh) 由 _get_global_id 给出
__kernel void one_tile(__global const half* Q, __global const half* Kt,
                       __global half* S, float scale, int ti, int tj, int bh) {
    __local half Qs[64][64];
    __local half Ks[64][64];
    int lid = get_local_id(1) * 16 + get_local_id(0);
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
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float acc = 0.f;
            for (int kk = 0; kk < 64; ++kk)
                acc += (float)Qs[r0 + i][kk] * (float)Ks[c0 + j][kk];
            Sp[(long)(ti * 64 + r0 + i) * 1024 + (tj * 64 + c0 + j)] = (half)(acc * scale);
        }
    }
}
"""
prog = cl.Program(ctx, KSRC).build(options=['-cl-mad-enable'])
kern = cl.Kernel(prog, 'one_tile')
mf = cl.mem_flags
rng = np.random.default_rng(0)
Bh, Th, HD = 4, 1024, 64
Qh = (rng.standard_normal((Bh, Th, HD)) * 0.5).astype(np.float16)
Kh = (rng.standard_normal((Bh, Th, HD)) * 0.5).astype(np.float16)
Qb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Qh)
Kb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Kh)
S = cl.Buffer(ctx, mf.READ_WRITE, Bh * Th * Th * 2)
z = np.zeros(Bh * Th * Th, np.float16)
cl.enqueue_copy(q_, S, z, is_blocking=True)
q_.finish()
SC = 1.0 / np.sqrt(HD)
for (ti, tj, bh) in ((0, 0, 0), (1, 2, 1), (3, 5, 2)):
    kern(q_, (1,), (256,), Qb, Kb, S, np.float32(SC), np.int32(ti), np.int32(tj), np.int32(bh))
q_.finish()
out = np.empty(Bh * Th * Th, np.float16)
cl.enqueue_copy(q_, out, S)
q_.finish()
out = out.reshape(Bh, Th, Th).astype(np.float32)
bad = 0
for (ti, tj, bh) in ((0, 0, 0), (1, 2, 1), (3, 5, 2)):
    blk = out[bh, ti * 64:(ti + 1) * 64, tj * 64:(tj + 1) * 64]
    ref = (Qh[bh, ti * 64:(ti + 1) * 64].astype(np.float32)
           @ Kh[bh, tj * 64:(tj + 1) * 64].astype(np.float32).T) * SC
    rel = float(np.abs(blk - ref).max() / (np.abs(ref).max() + 1e-12))
    print('tile (ti=%d,tj=%d,bh=%d): rel=%.3e %s' % (ti, tj, bh, rel, 'OK' if rel < 5e-3 else 'FAIL'))
    bad += rel > 5e-3
print('非零覆盖: %d / %d' % (int((out != 0).sum()), out.size))
