# -*- coding: utf-8 -*-
"""bench_tile_sweep.py — 扫 tile 配置，把算术强度从 16 推到 42

背景（已实测）：
  核显纯 FMA 算力:  963.5 GFLOPS  ← 没有频率墙（84% 理论值）
  核显带宽:          35.8 GB/s   ← 干净重测值（原写 27.6，已被报告 §10.79 撤回；口径见 §10.80）
  拐点 ridge point = 963.5/35.8 = 【26.9 FLOP/字节】
  （★ 2026-09-25 更正：原写 963.5/27.6 = 35。27.6 已被报告 §10.79 撤回 ——
    干净重测的核显单跑带宽是 35.8 GB/s，故 ridge = 26.9；口径见 §10.80。）

  而 64×64 tile 的算术强度 = TM·TN/(2(TM+TN))
    8×8  → 16 FLOP/字节  ← 远低于拐点 ⇒ 【带宽受限】
    ⇒ 上限 = 35.8 × 16 = 573，CLBlast 实测 278.6（49%）
      （★ 2026-09-25 更正：原写「27.6 × 16 = 442，63%」。27.6 已被报告 §10.79 撤回，
        干净重测的核显单跑带宽是 35.8 GB/s ⇒ 上限与占比都要跟着改，口径见 §10.80。）

  提高强度的唯一办法是【加大每个线程的计算块】：
    8×8  → 16
    8×16 → 21.3
    16×16→ 32
    16×32→ 42.7  ← 超过拐点，变算力受限

本脚本扫 (TS, TM, TN, TK) 组合，找最优。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import time
import numpy as np
import pyopencl as cl

KERNEL_TMPL = r'''
#define TS   @TS@
#define TM   @TM@
#define TN   @TN@
#define TK   @TK@
#define NTH  ((TS/TM) * (TS/TN))

__kernel void sgemm_t(const int M, const int N, const int K,
                      const __global float* restrict A,
                      const __global float* restrict B,
                      __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];

    const int tx = get_local_id(0);           // 0 .. TS/TN-1
    const int ty = get_local_id(1);           // 0 .. TS/TM-1
    const int bx = get_group_id(0) * TS;
    const int by = get_group_id(1) * TS;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // 载入两张 tile：TK*TS*2 个元素，NTH 个线程分担
        const int lane = ty * (TS/TN) + tx;
        #pragma unroll
        for (int p = 0; p < (TK * TS * 2) / (NTH * 4); p++) {
            int flat4 = p * NTH + lane;
            int total4 = TK * TS / 4;
            if (flat4 < total4) {
                int r = (flat4 * 4) / TS, c = (flat4 * 4) % TS;
                int gr = by + r, gc = k0 + c;
                float4 v = (gr < M && gc + 3 < K)
                    ? *(const __global float4*)(A + gr * K + gc) : (float4)0.f;
                As[r][c]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w;
            } else {
                int f2 = flat4 - total4;
                int r = (f2 * 4) / TS, c = (f2 * 4) % TS;
                int gr = k0 + r, gc = bx + c;
                float4 v = (gr < K && gc + 3 < N)
                    ? *(const __global float4*)(B + gr * N + gc) : (float4)0.f;
                Bs[r][c]=v.x; Bs[r][c+1]=v.y; Bs[r][c+2]=v.z; Bs[r][c+3]=v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
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
'''

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
print('=' * 82)
print('Tile 配置扫描 —— 把算术强度从 16 推到 42')
print('=' * 82)
print('  设备:', dev.name, '|', dev.max_compute_units, 'CU | local',
      dev.local_mem_size // 1024, 'KB')
print('  参照: 纯 FMA 963.5 GFLOPS | 带宽 35.8 GB/s | 拐点 26.9 FLOP/字节'
      '   （原写 27.6 / 35，口径见报告 §10.80）')
print()

# ---- 核显必须预热 20s+，否则数据偏低 78%（实测过的坑）----
import time as _t
print('  预热核显 25 秒...', flush=True)
_w = cl.Program(ctx, KERNEL_TMPL.replace('@TS@','64').replace('@TM@','8').replace('@TN@','8').replace('@TK@','16')).build()
_wk = cl.Kernel(_w, 'sgemm_t')
_WN = 512
_A = np.random.randn(_WN, _WN).astype(np.float32); _B = np.random.randn(_WN, _WN).astype(np.float32)
_mf = cl.mem_flags
_ab = cl.Buffer(ctx, _mf.READ_ONLY | _mf.COPY_HOST_PTR, hostbuf=_A)
_bb = cl.Buffer(ctx, _mf.READ_ONLY | _mf.COPY_HOST_PTR, hostbuf=_B)
_cb = cl.Buffer(ctx, _mf.WRITE_ONLY, _WN * _WN * 4)
_t0 = _t.perf_counter()
while _t.perf_counter() - _t0 < 25:
    for _ in range(8):
        _wk(queue, (_WN // 64 * 8, _WN // 64 * 8), (8, 8),
            np.int32(_WN), np.int32(_WN), np.int32(_WN), _ab, _bb, _cb)
    queue.finish()
del _ab, _bb, _cb, _A, _B
print('  预热完成\n', flush=True)

CONFIGS = [
    (64,  8,  8,  8, 'TK=8   barrier 256 次'),
    (64,  8,  8, 16, 'TK=16  barrier 128 次  ← 上一版最好'),
    (64,  8,  8, 32, 'TK=32  barrier  64 次'),
    (32,  8,  8, 32, 'TS=32 TK=32'),
    (32,  8,  8, 64, 'TS=32 TK=64  barrier 32 次'),
    (48,  8,  8, 32, 'TS=48 TK=32'),
]
results = []

for TS, TM, TN, TK, note in CONFIGS:
    nth = (TS // TM) * (TS // TN)
    if nth < 16 or nth > 1024:
        print('  跳过 TS=%d TM=%d TN=%d: 线程数 %d 不合理' % (TS, TM, TN, nth))
        continue
    intensity = TS / 4.0
    try:
        src = KERNEL_TMPL.replace('@TS@', str(TS)).replace('@TM@', str(TM)).replace('@TN@', str(TN)).replace('@TK@', str(TK))
        prg = cl.Program(ctx, src).build()
        kern = cl.Kernel(prg, 'sgemm_t')
    except Exception as e:
        print('  TS=%-4d TM=%-2d TN=%-2d: 编译失败 %s' % (TS, TM, TN, str(e)[:60]))
        continue

    row = []
    for n in (1024, 2048):
        M = N = K = n
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        mf = cl.mem_flags
        ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        gsz = ((N + TS - 1) // TS * (TS // TN), (M + TS - 1) // TS * (TS // TM))
        ls = (TS // TN, TS // TM)
        try:
            def one():
                kern(queue, gsz, ls, np.int32(M), np.int32(N), np.int32(K), ab, bb, cb)
                queue.finish()
            one()
            reps = 4
            t0 = time.perf_counter()
            for _ in range(reps):
                one()
            dt = (time.perf_counter() - t0) / reps
            gf = 2.0 * M * N * K / dt / 1e9
            row.append(gf)
        except Exception as e:
            row.append(0.0)
        del A, B, ab, bb, cb

    g1024, g2048 = row[0], row[1]
    best = max(row)
    # ★★ 2026-09-25 修正：带宽 27.6 → 35.8 GB/s（27.6 已被报告 §10.79 撤回）。
    #   这一处【是计算，不是显示】：lim 是这张 roofline 表的带宽上限，
    #   表里每一行的「(带宽上限 X 的 Y%)」都由它算出 ⇒ 用旧值会把所有行都算错，
    #   而且方向是【低估上限】⇒ 看起来"更接近饱和"，可能误判 tile 已经够好。
    lim = min(963.5, 35.8 * intensity)
    print('  TS=%-4d TM=%-2d TN=%-2d TK=%-2d 线程=%-4d 强度=%5.1f  '
          '1024³=%7.1f  2048³=%7.1f  GFLOPS  (带宽上限 %.0f 的 %.0f%%)  %s'
          % (TS, TM, TN, TK, nth, intensity, g1024, g2048, lim,
             100 * g2048 / lim if lim else 0, note))
    results.append((best, TS, TM, TN, TK, intensity, g2048))

print()
print('=' * 82)
print('对照表')
print('=' * 82)
print('  纯 FMA 算力上限           963.5 GFLOPS')
print('  DML（AMD 官方）            294.5')
print('  CLBlast（业界标准）         278.6')
print('  我上一版调优 kernel         245.3')
if results:
    results.sort(reverse=True)
    b = results[0]
    print('  【本次最佳  TS=%d TM=%d TN=%d TK=%d】  %.1f GFLOPS  (强度 %.1f)'
          % (b[1], b[2], b[3], b[4], b[0], b[5]))
    print()
    print('  ⇒ 相对上一版: %.2fx   相对 CLBlast: %.2fx   相对 DML: %.2fx'
          % (b[0] / 245.3, b[0] / 278.6, b[0] / 294.5))
print('DONE')
