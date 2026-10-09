# -*- coding: utf-8 -*-
"""bench_opencl3.py — 向 CLBlast 级别冲击的 SGEMM v2

上一版（bench_opencl2.py）瓶颈定位：
  内层循环每 64 个 FMA 要配 16 次【标量】LDS 读
  GCN 的 LDS 是每 CU 共享 128 B/周期 → 这就是 FMA 单元 79% 空转的来源

本版改进：
  1) LDS 用 float4 读：16 次标量 → 4 次向量
     （padding 用 +4 而不是 +1，保证 16 字节对齐：
       行距 (TS+4)*4 = 272 B = 17×16 ✅）
  2) 全局→LDS 载入也改 float4（上一版已有）
  3) 加 TS=128 变体：本地内存 32KB 够用，更大的 tile = 更高数据复用

对照基线：
  DML            294.5 GFLOPS (26%)   ← AMD 自己的路径
  上一版自写      245.3 GFLOPS (21%)
  理论峰值       1150 GFLOPS
"""
import time
import numpy as np
import pyopencl as cl

platform = cl.get_platforms()[0]
dev = platform.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)
BATCH = 8      # 队列深度上限（上一版 bug：不限流会永远跑不完）

print('=' * 76)
print('SGEMM v2  |  device:', dev.name, '|', dev.max_compute_units, 'CU  |  local',
      dev.local_mem_size // 1024, 'KB')
print('=' * 76)

# --------------------------- kernel 源码 ---------------------------
SRC = r'''
// ============ v2a: TS=64, TK=16, float4 LDS ============
#define TS 64
#define TK 16
#define TM 8
#define TN 8

__kernel void sgemm_v2a(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    // +4 padding：行距 (TS+4)*4 = 272B = 17*16，保证 float4 对齐
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];

    const int tx = get_local_id(0);      // 0..7
    const int ty = get_local_id(1);      // 0..7
    const int bx = get_group_id(0) * TS;
    const int by = get_group_id(1) * TS;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        const int lane = ty * 8 + tx;
        #pragma unroll
        for (int p = 0; p < (TK * TS * 2) / (64 * 4); p++) {
            int flat4 = p * 64 + lane;
            int total4 = TK * TS / 4;
            if (flat4 < total4) {
                int r = (flat4 * 4) / TS, c = (flat4 * 4) % TS;
                int gr = by + r, gc = k0 + c;
                float4 v = (gr < M && gc + 3 < K)
                    ? *(const __global float4*)(A + gr * K + gc) : (float4)0.f;
                As[r][c] = v.x; As[r][c+1] = v.y; As[r][c+2] = v.z; As[r][c+3] = v.w;
            } else {
                int f2 = flat4 - total4;
                int r = (f2 * 4) / TS, c = (f2 * 4) % TS;
                int gr = k0 + r, gc = bx + c;
                float4 v = (gr < K && gc + 3 < N)
                    ? *(const __global float4*)(B + gr * N + gc) : (float4)0.f;
                Bs[r][c] = v.x; Bs[r][c+1] = v.y; Bs[r][c+2] = v.z; Bs[r][c+3] = v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TK; k++) {
            // ---- 关键改动：float4 从 LDS 读 ----
            float4 a0 = *(const __local float4*)&As[k][ty * TM + 0];
            float4 a1 = *(const __local float4*)&As[k][ty * TM + 4];
            float4 b0 = *(const __local float4*)&Bs[k][tx * TN + 0];
            float4 b1 = *(const __local float4*)&Bs[k][tx * TN + 4];
            float a[TM] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[TN] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int r = by + ty * TM + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int c = bx + tx * TN + j;
            if (c < N) C[r * N + c] = acc[i][j];
        }
    }
}

// ============ v2b: TS=128, TK=8, float4 LDS ============
// 本地内存：2 × 8 × 132 × 4 = 8.4 KB（32KB 上限内）
// 更大的 tile = 全局访存复用更高
#define TS2 128
#define TK2 8
#define TM2 8
#define TN2 8

__kernel void sgemm_v2b(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TK2][TS2 + 4];
    __local float Bs[TK2][TS2 + 4];

    const int tx = get_local_id(0);      // 0..15
    const int ty = get_local_id(1);      // 0..15
    const int bx = get_group_id(0) * TS2;
    const int by = get_group_id(1) * TS2;

    float acc[TM2][TN2];
    #pragma unroll
    for (int i = 0; i < TM2; i++)
        #pragma unroll
        for (int j = 0; j < TN2; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK2) {
        // 128x8 两张 tile，共 2048 元素 = 512 个 float4，256 线程分担
        const int lane = ty * 16 + tx;
        #pragma unroll
        for (int p = 0; p < 2; p++) {
            int flat4 = p * 256 + lane;
            int total4 = TK2 * TS2 / 4;
            if (flat4 < total4) {
                int r = (flat4 * 4) / TS2, c = (flat4 * 4) % TS2;
                int gr = by + r, gc = k0 + c;
                float4 v = (gr < M && gc + 3 < K)
                    ? *(const __global float4*)(A + gr * K + gc) : (float4)0.f;
                As[r][c] = v.x; As[r][c+1] = v.y; As[r][c+2] = v.z; As[r][c+3] = v.w;
            } else {
                int f2 = flat4 - total4;
                int r = (f2 * 4) / TS2, c = (f2 * 4) % TS2;
                int gr = k0 + r, gc = bx + c;
                float4 v = (gr < K && gc + 3 < N)
                    ? *(const __global float4*)(B + gr * N + gc) : (float4)0.f;
                Bs[r][c] = v.x; Bs[r][c+1] = v.y; Bs[r][c+2] = v.z; Bs[r][c+3] = v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TK2; k++) {
            float4 a0 = *(const __local float4*)&As[k][ty * TM2 + 0];
            float4 a1 = *(const __local float4*)&As[k][ty * TM2 + 4];
            float4 b0 = *(const __local float4*)&Bs[k][tx * TN2 + 0];
            float4 b1 = *(const __local float4*)&Bs[k][tx * TN2 + 4];
            float a[TM2] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[TN2] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < TM2; i++)
                #pragma unroll
                for (int j = 0; j < TN2; j++)
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int i = 0; i < TM2; i++) {
        int r = by + ty * TM2 + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN2; j++) {
            int c = bx + tx * TN2 + j;
            if (c < N) C[r * N + c] = acc[i][j];
        }
    }
}
'''

try:
    prg = cl.Program(ctx, SRC).build()
    K2A = cl.Kernel(prg, 'sgemm_v2a')
    K2B = cl.Kernel(prg, 'sgemm_v2b')
    print('  两个 kernel 都编译通过')
except Exception as e:
    print('  编译失败:')
    print(str(e)[:3000])
    raise SystemExit(1)


def run(kern, TS, gmul, M, N, K, iters):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    mf = cl.mem_flags
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + TS - 1) // TS * gmul, (M + TS - 1) // TS * gmul)
    args = (np.int32(M), np.int32(N), np.int32(K), ab, bb, cb)

    def batch(nb):
        for _ in range(nb):
            kern(queue, gsz, (gmul, gmul), *args)
        queue.finish()

    batch(BATCH)                     # 预热（核显必须预热，否则数据偏低 78%）
    t0 = time.perf_counter()
    batch(BATCH * 2)
    el = time.perf_counter() - t0

    # 回读校验：这个脚本原先只计时、不看结果，所以一个"跑得快但算错"的内核
    # 会给出完全一样的 GFLOPS。CLBlast 在同一台设备上正是如此（644.9 GFLOPS，
    # 相对误差 1.3~1.6），因此这里把正确性和计时放在同一次运行里读出来。
    out = np.empty((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, out, cb).wait()
    ref = A @ B
    alt = B @ A                                  # 方阵下两种顺序都是合法 GEMM
    denom = float(np.abs(ref).max()) or 1.0
    maxdiff = float(np.abs(out - ref).max())
    maxdiff_ba = float(np.abs(out - alt).max())
    # 只要与其中一种吻合，内核就是在做真实的 GEMM —— 计算量相同，吞吐量可用。
    # 对角阵与单位阵都无法区分 A、B 的读取顺序（行列全同），必须用非对称输入。
    rel = min(maxdiff, maxdiff_ba) / denom
    order = 'A@B' if maxdiff <= maxdiff_ba else 'B@A（需对调传参）'

    return BATCH * 2 * 2 * M * N * K / el / 1e9, el, maxdiff, rel, order


print()
print('  预热核显 10 秒...', flush=True)
t0 = time.perf_counter()
A0 = np.random.randn(1024, 1024).astype(np.float32)
B0 = np.random.randn(1024, 1024).astype(np.float32)
mf = cl.mem_flags
a0 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A0)
b0 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B0)
c0 = cl.Buffer(ctx, mf.WRITE_ONLY, 1024 * 1024 * 4)
while time.perf_counter() - t0 < 10:
    for _ in range(8):
        K2A(queue, (1024 // 64 * 8, 1024 // 64 * 8), (8, 8),
            np.int32(1024), np.int32(1024), np.int32(1024), a0, b0, c0)
    queue.finish()
del a0, b0, c0, A0, B0
print('  预热完成\n', flush=True)

print('  %-22s %10s %12s %10s' % ('配置', 'n', 'GFLOPS', '占理论%'), flush=True)
best = {}
for tag, kern, TS, gmul in (('v2a TS=64  8x8', K2A, 64, 8),
                            ('v2b TS=128 16x16', K2B, 128, 16)):
    for n in (512, 1024, 2048):
        try:
            gf, el, maxdiff, rel, order = run(kern, TS, gmul, n, n, n, 4)
            best[(tag, n)] = gf
            print('  %-22s %10d %12.2f %9.1f%%   正确性: 相对 %.1e 约定=%s %s'
                  % (tag, n, gf, 100 * gf / 1150, rel, order,
                     '✓' if rel < 1e-4 else '✗✗ 结果错'), flush=True)
        except Exception as e:
            print('  %-22s %10d 失败: %s' % (tag, n, str(e)[:50]), flush=True)
    print()

print('=== 对照 ===')
print('  理论峰值           1150 GFLOPS')
print('  DML（AMD 自己）     294.5 GFLOPS (26%)')
print('  上一版自写          245.3 GFLOPS (21%)')
for (tag, n), gf in sorted(best.items(), key=lambda kv: -kv[1])[:3]:
    print('  本次最佳  %-18s n=%-5d %8.2f GFLOPS (%.1f%%)' % (tag, n, gf, 100 * gf / 1150))
print()
print('DONE', flush=True)
