"""Redo the concurrency test with processes instead of threads. The 0.76x may be a GIL artefact.

The arithmetic that forced this. A GT 1030 does roughly 600-700 GFLOPS on GEMM with its own 48 GB/s
of memory. We have 240 GFLOPS from the CPU alone and 361.5 from the swizzled iGPU kernel -- 601.5
together, which is exactly the 1030's range. So matching a 1030 needs no new hardware, only that the
two engines stop interfering. The obstacle is that concurrency measured 0.76x of the CPU alone.

Then the second piece of arithmetic, which says the interference is not memory. A GEMM at these
shapes has an arithmetic intensity near 300 FLOP per byte, so the CPU's 240 GFLOPS needs 0.8 GB/s and
the iGPU's 361.5 needs 1.2. That is 2 GB/s against the 27 GB/s this machine measured, seven percent
of the bus. Bandwidth cannot be what is costing 24%.

Which leaves three candidates: L3 contention, the 15 W power budget, or the measurement itself.

The third is the one to eliminate first, and it is cheap. The earlier test ran the GPU in a Python
thread and the CPU in the main thread. Every kernel launch from that thread needs the GIL, torch
releases it during heavy compute but reacquires between operators, and the two threads therefore
queue on the GIL. That queueing would appear in the timing as contention, and it is not contention
at all.

So: same measurement, same kernel, same shapes, but the GPU side in a separate process. If the
answer moves toward 1.5x, the "iGPU is closed" conclusion is wrong and we already have the
equivalent of a discrete card. If it stays near 0.76x, the limit is real, and the clock sampled
during the run will say whether it is power.

Decision rule, fixed in advance: 1.2x or better means the iGPU path reopens, exactly as 7.46
specified. Below that it stays closed, but this time on evidence that has ruled out the GIL.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import multiprocessing as mp
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


SRC = r'''
#define TS 64
#define TK 16
#define TM 4
#define TN 4
__kernel void gemm_fast(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int bx = get_group_id(0) * TS;
    const int by = get_group_id(1) * TS;
    const int lane = ty * 16 + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        {
            int f4 = lane;
            int r = (f4 * 4) / TK, c = (f4 * 4) % TK;
            int gr = by + r, gc = k0 + c;
            float4 v = (float4)0.f;
            if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr * K + gc);
            As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w;
        }
        {
            int f4 = lane;
            int k = (f4 * 4) / TS, n = (f4 * 4) % TS;
            int gk = k0 + k, gn = bx + n;
            float4 v = (float4)0.f;
            if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk * N + gn);
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int cc = n + q;
                Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                    (q==0)?v.x:((q==1)?v.y:((q==2)?v.z:v.w));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j * (TS / TN) + tx];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = by + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gc = bx + tx * TN + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}
'''

SHAPES = [(2048, 512, 512), (2048, 2736, 512), (2048, 512, 2736), (2048, 2048, 512)]
FLOP = sum(2.0 * M * N * K for (M, N, K) in SHAPES)
SECONDS = 25.0


def gpu_worker(rounds_target, queue):
    """Runs in its own process, so no GIL interaction with the CPU side at all."""
    import pyopencl as cl
    plat = cl.get_platforms()[0]
    ctx = cl.Context(devices=[plat.get_devices()[0]])
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    rng = np.random.default_rng(11)
    bufs = []
    for (M, N, K) in SHAPES:
        A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
        Bm = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
        bufs.append((M, N, K,
                     cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A),
                     cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Bm),
                     cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)))
    kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'gemm_fast')

    def one_group():
        for (M, N, K, ba, bb, bc) in bufs:
            kern(q, ((N + 63) // 64 * 16, (M + 63) // 64 * 16), (16, 16),
                 np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)

    one_group()
    q.finish()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < rounds_target:
        one_group()
        n += 1
    q.finish()
    el = time.perf_counter() - t0
    queue.put(('gpu', n, el, el / max(n, 1)))


def main():
    from measure_guard import assert_idle, stamp
    assert_idle()
    stamp('并发重测：进程而非线程（排除 GIL）')
    torch.set_num_threads(6)
    tok = torch.load(r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt', map_location='cpu')

    B, T = 2, 1024
    from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    from loop_transformer import LoopConfig
    cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
                             ffn_dim=2048, total_tokens=T, grid=32, head_mode='flat',
                             mask_mode='causal', n_cluster=32, use_bos=False)
    model = SmallARImageModelV2(cfg)
    model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end, loop_L=1,
                              loop_L_max=3, loop_random=False, loop_cache='shared'))
    model.train()
    print('  模型 %.1fM 参数 · 设备 %s' % (sum(p.numel() for p in model.parameters()) / 1e6,
                                       __import__('pyopencl').get_platforms()[0].get_devices()[0].name))
    print()

    def cpu_steps(secs):
        g = torch.Generator().manual_seed(0)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < secs:
            x = tok[torch.randint(0, len(tok), (B,), generator=g)].long()
            out = model.forward_logits(x, None)
            lg = out[1]
            lg = lg[0] if isinstance(lg, tuple) else lg
            torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size),
                                              x[:, 1:].reshape(-1)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            n += 1
        el = time.perf_counter() - t0
        return n, el

    print('--- CPU 单独（%.0f 秒）---' % SECONDS)
    n1, e1 = cpu_steps(SECONDS)
    print('    %d 步 · %.3f s/步' % (n1, e1 / n1))

    print('--- 核显单独（%.0f 秒，独立进程）---' % SECONDS)
    qq = mp.Queue()
    p = mp.Process(target=gpu_worker, args=(SECONDS, qq))
    p.start()
    tag, gn, gel, gper = qq.get()
    p.join()
    print('    %d 组 · %.4f s/组' % (gn, gper))

    print('--- 并发：核显在【独立进程】，CPU 在【本进程】---')
    q2 = mp.Queue()
    p2 = mp.Process(target=gpu_worker, args=(SECONDS + 5, q2))
    p2.start()
    time.sleep(3)
    n2, e2 = cpu_steps(SECONDS)
    tag2, gn2, gel2, gper2 = q2.get()
    p2.join()
    print('    CPU %d 步 · %.3f s/步' % (n2, e2 / n2))
    print()
    print('=' * 64)
    print('  CPU 单独        %.3f s/步' % (e1 / n1))
    print('  并发时 CPU      %.3f s/步  ⇒ 【慢 %.1f%%】' % (e2 / n2, (e2 / n2) / (e1 / n1) * 100 - 100))
    print('  核显单独        %.4f s/组  (%.1f GFLOPS)' % (gper, 2 * FLOP / gper / 1e9))
    print('  并发时核显      %.4f s/组  (%.1f GFLOPS)' % (gper2, 2 * FLOP / gper2 / 1e9))
    tot_alone = 2 * FLOP / gper / 1e9 + 240.0
    print('=' * 64)
    print()
    print('  判据（7.46 事先固定）: 并发 ≥ CPU 单独 1.2 倍 ⇒ 核显重开')
    print('  若这次结果明显好于线程版的 0.76× ⇒ ★ 那个数字是 GIL 造成的伪影')
    print('  参考: GT 1030 约 600~700 GFLOPS ⇒ CPU 240 + 核显 361.5 = 601.5,【正好是 1030 的量级】')


if __name__ == '__main__':
    mp.freeze_support()
    main()
