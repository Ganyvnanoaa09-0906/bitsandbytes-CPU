# -*- coding: utf-8 -*-
"""编译 ar_v2_multi 里的新 kernel 并打印 build log（定位 OpenCL 语法错误）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import pyopencl as cl
import gt1030.attention_gpu as A

SRC = r"""
#define TS 64
#define HDx 64
__kernel void s_gemm(__global const half* Q, __global const half* Kt,
                     __global half* S, float scale) {
    int bh = get_global_id(2);
    int ti = get_global_id(0);
    int tj = get_global_id(1);
    __local half Qs[64][64];
    __local half Ks[64][64];
    int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int lx = get_local_id(0), ly = get_local_id(1);
    __global const half* Qp = Q + (long)bh * 1024 * HDx;
    __global const half* Kp = Kt + (long)bh * 1024 * HDx;
    __global half* Sp = S + (long)bh * 1024 * 1024;
    for (int i = lid; i < TS * TS; i += 256) {
        int r = i / TS, c = i % TS;
        Qs[r][c] = Qp[(long)(ti * TS + r) * HDx + c];
        Ks[r][c] = Kp[(long)(tj * TS + r) * HDx + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int r0 = ly * 4, c0 = lx * 4;
    float acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int kk = 0; kk < HDx; ++kk) {
        float qv[4], kv[4];
        for (int i = 0; i < 4; ++i) qv[i] = (float)Qs[r0 + i][kk];
        for (int j = 0; j < 4; ++j) kv[j] = (float)Ks[c0 + j][kk];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) acc[i][j] += qv[i] * kv[j];
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Sp[(long)(ti * TS + r0 + i) * 1024 + (tj * TS + c0 + j)] = (half)(acc[i][j] * scale);
}
"""
try:
    prog = cl.Program(A.ctx, SRC).build(options=['-cl-mad-enable'])
    print('build OK')
except Exception as e:
    log = getattr(e, 'build_log', None) or str(e)
    print('BUILD FAILED:\n', log[:3000] if isinstance(log, (bytes, str)) else log)
