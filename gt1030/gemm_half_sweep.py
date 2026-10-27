# -*- coding: utf-8 -*-
"""参数化扫 fp16 GEMM 配置（块 tile / 每线程寄存器块 / BK / 向量宽度），找打赢 CLBlast 的点。

用法：python gt1030/gemm_half_sweep.py
"""
from __future__ import annotations
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl

SRC_TMPL = r"""
#define MS {MS}
#define NS {NS}
#define TM {TM}
#define TN {TN}
#define LX {LX}
#define LY {LY}
#define BK {BK}
#define NTH (LX*LY)
#define VW {VW}

__kernel void gemm_nn_h(__global const half* A, __global const half* B, __global half* C,
                        int M, int N, int K, float alpha) {
    __local half Al[MS * BK];
    __local half Bl[BK * NS];
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int mb = get_group_id(1) * MS;
    int nb = get_group_id(0) * NS;
    int tid = ly * LX + lx;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int t = tid; t < MS * BK / VW; t += NTH) {
            int r = t / (BK / VW);
            int c = (t % (BK / VW)) * VW;
            int gm = mb + r, gk = k0 + c;
            if (VW == 8) {
                half8 v = (half8)(0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f);
                if (gm < M && gk + 7 < K) v = vload8(0, A + (long)gm * K + gk);
                vstore8(v, 0, (__local half*)Al + r * BK + c);
            } else {
                half4 v = (half4)(0.f,0.f,0.f,0.f);
                if (gm < M && gk + 3 < K) v = vload4(0, A + (long)gm * K + gk);
                vstore4(v, 0, (__local half*)Al + r * BK + c);
            }
        }
        for (int t = tid; t < BK * NS / VW; t += NTH) {
            int r = t / (NS / VW);
            int c = (t % (NS / VW)) * VW;
            int gk = k0 + r, gn = nb + c;
            if (VW == 8) {
                half8 v = (half8)(0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f);
                if (gk < K && gn + 7 < N) v = vload8(0, B + (long)gk * N + gn);
                vstore8(v, 0, (__local half*)Bl + r * NS + c);
            } else {
                half4 v = (half4)(0.f,0.f,0.f,0.f);
                if (gk < K && gn + 3 < N) v = vload4(0, B + (long)gk * N + gn);
                vstore4(v, 0, (__local half*)Bl + r * NS + c);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        __local half* ap = (__local half*)Al + (ly * TM) * BK;
        for (int kk = 0; kk < BK; ++kk) {
            float av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) av[i] = (float)ap[i * BK + kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) bv[j] = (float)Bl[kk * NS + lx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += av[i] * bv[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < TM; ++i) {
        int gm = mb + ly * TM + i;
        if (gm >= M) break;
        for (int j = 0; j < TN; ++j) {
            int gn = nb + lx * TN + j;
            if (gn >= N) break;
            C[(long)gm * N + gn] = (half)(acc[i][j] * alpha);
        }
    }
}
"""

CFGS = [
    dict(name='64x64 t4x4 16x16 BK32 VW4', MS=64, NS=64, TM=4, TN=4, LX=16, LY=16, BK=32, VW=4),
    dict(name='64x64 t4x4 16x16 BK16 VW4', MS=64, NS=64, TM=4, TN=4, LX=16, LY=16, BK=16, VW=4),
    dict(name='64x64 t4x4 16x16 BK64 VW8', MS=64, NS=64, TM=4, TN=4, LX=16, LY=16, BK=64, VW=8),
    dict(name='64x64 t8x8 8x8   BK32 VW8', MS=64, NS=64, TM=8, TN=8, LX=8, LY=8, BK=32, VW=8),
    dict(name='128x64 t8x4 16x16 BK32 VW8', MS=128, NS=64, TM=8, TN=4, LX=16, LY=16, BK=32, VW=8),
    dict(name='64x128 t4x8 16x16 BK32 VW8', MS=64, NS=128, TM=4, TN=8, LX=16, LY=16, BK=32, VW=8),
    dict(name='128x128 t8x8 16x16 BK32 VW8', MS=128, NS=128, TM=8, TN=8, LX=16, LY=16, BK=32, VW=8),
    dict(name='128x64 t8x4 16x16 BK64 VW8', MS=128, NS=64, TM=8, TN=4, LX=16, LY=16, BK=64, VW=8),
    dict(name='64x64 t4x4 16x16 BK32 VW8', MS=64, NS=64, TM=4, TN=4, LX=16, LY=16, BK=32, VW=8),
    dict(name='128x128 t8x8 16x16 BK16 VW8', MS=128, NS=128, TM=8, TN=8, LX=16, LY=16, BK=16, VW=8),
]


def main():
    import gt1030.ar_v2_multi as MM
    import gt1030.attention_gpu as A
    import gt1030.full_gpu_layer as L

    ctx, q = A.ctx, A.queue
    mf = cl.mem_flags
    M = 2048
    shapes = [('Wqkv', 1024, 512), ('Wo', 512, 512), ('W12', 2736, 512), ('W3', 512, 1368)]
    rng = np.random.default_rng(0)
    data = {}
    for nm, n, k in shapes:
        a = (rng.standard_normal((M, k)) * 0.05).astype(np.float16)
        b = (rng.standard_normal((k, n)) * 0.05).astype(np.float16)
        data[nm] = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
                    cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b),
                    cl.Buffer(ctx, mf.READ_WRITE, M * n * 2),
                    cl.Buffer(ctx, mf.READ_WRITE, M * n * 2), n, k, a, b)

    def timeit(fn, n=8):
        fn(); q.finish()
        t0 = time.time()
        for _ in range(n):
            fn()
        q.finish()
        return (time.time() - t0) / n

    # CLBlast 基线
    base = {}
    for nm, n, k in shapes:
        ab, bb, c1, c2, n_, k_, a, b = data[nm]
        dt = timeit(lambda: L.hgemm_nn(ab, bb, c1, M, n_, k_))
        base[nm] = 2.0 * M * n_ * k_ / dt / 1e9
    print('CLBlast 基线 GFLOPS: ' + '  '.join('%s=%.0f' % kv for kv in base.items()))
    print()
    best = None
    for cfg in CFGS:
        src = SRC_TMPL
        for k, v in cfg.items():
            if k != 'name':
                src = src.replace('{%s}' % k, str(v))
        try:
            prog = cl.Program(ctx, src).build(options=['-cl-mad-enable'])
            kern = cl.Kernel(prog, 'gemm_nn_h')
        except Exception as e:
            print('%-32s BUILD FAIL %s' % (cfg['name'], str(e)[:60])); continue
        rows, tot = [], 0.0
        ok = True
        for nm, n, k in shapes:
            ab, bb, c1, c2, n_, k_, a, b = data[nm]
            gx = (n_ + cfg['NS'] - 1) // cfg['NS']
            gy = (M + cfg['MS'] - 1) // cfg['MS']

            def run():
                kern(q, (gx * cfg['LX'], gy * cfg['LY']), (cfg['LX'], cfg['LY']),
                     ab, bb, c2, np.int32(M), np.int32(n_), np.int32(k_), np.float32(1.0))
            try:
                dt = timeit(run)
            except Exception as e:
                print('%-32s RUN FAIL %s' % (cfg['name'], str(e)[:50])); ok = False; break
            gf = 2.0 * M * n_ * k_ / dt / 1e9
            rows.append('%s=%.0f(%.2fx)' % (nm, gf, gf / base[nm]))
            tot += gf / base[nm]
        if not ok:
            continue
        # 数值抽查（用最后一个形状）
        ab, bb, c1, c2, n_, k_, a, b = data[shapes[-1][0]]
        o1 = np.empty((M, n_), np.float16); cl.enqueue_copy(q, o1, c1)
        o2 = np.empty((M, n_), np.float16); cl.enqueue_copy(q, o2, c2)
        q.finish()
        rel = float(np.abs(o2.astype(np.float32) - o1.astype(np.float32)).max()
                    / (np.abs(o1.astype(np.float32)).max() + 1e-12))
        score = tot / len(shapes)
        print('%-32s %s  avg=%.2fx rel=%.1e' % (cfg['name'], ' '.join(rows), score, rel))
        if best is None or score > best[0]:
            best = (score, cfg['name'])
    print()
    print('BEST:', best)


if __name__ == '__main__':
    main()
