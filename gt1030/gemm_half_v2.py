# -*- coding: utf-8 -*-
"""向量化 fp16 GEMM 第二版：A tile 转置进 local，使计算循环能用 half4 向量读 local。

上一版的问题：计算循环里每 kk 要做 TM+TN=8 次 **标量** local 读（16 次 FMA），
GCN 的 LDS 指令吞吐成了瓶颈（每 warp 一条 LDS 要 ~4 拍）。这一版：
  · A 在 global→local 时转置成 At[BK][MS]，于是 av = vload4(At + kk*MS + m0) 一条指令取 4 个
  · B 仍是 Bl[BK][NS]，bv = vload4(Bl + kk*NS + n0) 一条指令取 4 个
  · 每 kk 只需 2 条 LDS 指令换 16 次 FMA（TM=TN=4）；TM=TN=8 时 4 条换 64 次
bank 分析：同 ly 的 16 个线程读同一地址（广播），不同 ly 相隔 TM 个 half = 8 字节，
16 个 ly 正好铺满 32 个 bank，无冲突。
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

__kernel void gemm_nn_h(__global const half* A, __global const half* B, __global half* C,
                        int M, int N, int K, float alpha) {
    __local half At[BK * MS];      // 转置后的 A
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
        // A: 读 (m,k) 连续 half4，写到 At[k][m]（每个 half 单独落位）
        for (int t = tid; t < MS * BK / 4; t += NTH) {
            int r = t / (BK / 4);
            int c4 = (t % (BK / 4)) * 4;
            int gm = mb + r, gk = k0 + c4;
            half4 v = (half4)(0.f, 0.f, 0.f, 0.f);
            if (gm < M && gk + 3 < K) v = vload4(0, A + (long)gm * K + gk);
            At[(c4 + 0) * MS + r] = v.x;
            At[(c4 + 1) * MS + r] = v.y;
            At[(c4 + 2) * MS + r] = v.z;
            At[(c4 + 3) * MS + r] = v.w;
        }
        // B: 读 (k,n) 连续 half4，直接落位（行内连续）
        for (int t = tid; t < BK * NS / 4; t += NTH) {
            int r = t / (NS / 4);
            int c4 = (t % (NS / 4)) * 4;
            int gk = k0 + r, gn = nb + c4;
            half4 v = (half4)(0.f, 0.f, 0.f, 0.f);
            if (gk < K && gn + 3 < N) v = vload4(0, B + (long)gk * N + gn);
            vstore4(v, 0, (__local half*)Bl + r * NS + c4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int m0 = ly * TM, n0 = lx * TN;
        for (int kk = 0; kk < BK; ++kk) {
            float av[TM], bv[TN];
#pragma unroll
            for (int t = 0; t < TM; t += 4) {
                half4 a = vload4(0, (__local half*)At + kk * MS + m0 + t);
                av[t] = (float)a.x; av[t+1] = (float)a.y;
                av[t+2] = (float)a.z; av[t+3] = (float)a.w;
            }
#pragma unroll
            for (int t = 0; t < TN; t += 4) {
                half4 b = vload4(0, (__local half*)Bl + kk * NS + n0 + t);
                bv[t] = (float)b.x; bv[t+1] = (float)b.y;
                bv[t+2] = (float)b.z; bv[t+3] = (float)b.w;
            }
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
    dict(name='64x64 t4x4 BK32', MS=64, NS=64, TM=4, TN=4, LX=16, LY=16, BK=32),
    dict(name='64x64 t4x4 BK16', MS=64, NS=64, TM=4, TN=4, LX=16, LY=16, BK=16),
    dict(name='128x64 t8x4 BK32', MS=128, NS=64, TM=8, TN=4, LX=16, LY=16, BK=32),
    dict(name='128x64 t8x4 BK16', MS=128, NS=64, TM=8, TN=4, LX=16, LY=16, BK=16),
    dict(name='64x128 t4x8 BK32', MS=64, NS=128, TM=4, TN=8, LX=16, LY=16, BK=32),
    dict(name='128x128 t8x8 BK16', MS=128, NS=128, TM=8, TN=8, LX=16, LY=16, BK=16),
    dict(name='64x64 t8x8 BK32 8x8', MS=64, NS=64, TM=8, TN=8, LX=8, LY=8, BK=32),
    dict(name='128x128 t8x8 BK32', MS=128, NS=128, TM=8, TN=8, LX=16, LY=16, BK=32),
    dict(name='256x64 t16x4 BK16', MS=256, NS=64, TM=16, TN=4, LX=16, LY=16, BK=16),
    dict(name='256x64 t16x4 BK32', MS=256, NS=64, TM=16, TN=4, LX=16, LY=16, BK=32),
    dict(name='256x128 t16x8 BK16', MS=256, NS=128, TM=16, TN=8, LX=16, LY=16, BK=16),
    dict(name='512x64 t32x4 BK16', MS=512, NS=64, TM=32, TN=4, LX=16, LY=16, BK=16),
]


def build(ctx, cfg):
    src = SRC_TMPL
    for k, v in cfg.items():
        if k != 'name':
            src = src.replace('{%s}' % k, str(v))
    prog = cl.Program(ctx, src).build(options=['-cl-mad-enable'])
    return cl.Kernel(prog, 'gemm_nn_h')


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
        data[nm] = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
                    cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b),
                    cl.Buffer(ctx, mf.READ_WRITE, M * n * 2),
                    cl.Buffer(ctx, mf.READ_WRITE, M * n * 2), n, k]

    def timeit(fn, n=8):
        fn(); q.finish()
        t0 = time.time()
        for _ in range(n):
            fn()
        q.finish()
        return (time.time() - t0) / n

    base = {}
    for nm, n, k in shapes:
        ab, bb, c1, c2, n_, k_ = data[nm]
        dt = timeit(lambda: L.hgemm_nn(ab, bb, c1, M, n_, k_))
        base[nm] = 2.0 * M * n_ * k_ / dt / 1e9
    print('CLBlast: ' + '  '.join('%s=%.0f' % kv for kv in base.items()))
    best = None
    for cfg in CFGS:
        try:
            kern = build(ctx, cfg)
        except Exception as e:
            print('%-22s BUILD FAIL %s' % (cfg['name'], str(e)[:70])); continue
        rows, score, ok = [], 0.0, True
        for nm, n, k in shapes:
            ab, bb, c1, c2, n_, k_ = data[nm]
            gx = (n_ + cfg['NS'] - 1) // cfg['NS']
            gy = (M + cfg['MS'] - 1) // cfg['MS']

            def run():
                kern(q, (gx * cfg['LX'], gy * cfg['LY']), (cfg['LX'], cfg['LY']),
                     ab, bb, c2, np.int32(M), np.int32(n_), np.int32(k_), np.float32(1.0))
            try:
                dt = timeit(run)
            except Exception as e:
                print('%-22s RUN FAIL %s' % (cfg['name'], str(e)[:50])); ok = False; break
            gf = 2.0 * M * n_ * k_ / dt / 1e9
            rows.append('%s=%.0f(%.2fx)' % (nm, gf, gf / base[nm]))
            score += gf / base[nm]
        if not ok:
            continue
        ab, bb, c1, c2, n_, k_ = data[shapes[-1][0]]
        o1 = np.empty((M, n_), np.float16); cl.enqueue_copy(q, o1, c1)
        o2 = np.empty((M, n_), np.float16); cl.enqueue_copy(q, o2, c2)
        q.finish()
        rel = float(np.abs(o2.astype(np.float32) - o1.astype(np.float32)).max()
                    / (np.abs(o1.astype(np.float32)).max() + 1e-12))
        sc = score / len(shapes)
        print('%-22s %s  avg=%.2fx rel=%.1e' % (cfg['name'], ' '.join(rows), sc, rel))
        if best is None or sc > best[0]:
            best = (sc, cfg['name'])
    print('\nBEST: %.2fx  %s' % best)


if __name__ == '__main__':
    main()
