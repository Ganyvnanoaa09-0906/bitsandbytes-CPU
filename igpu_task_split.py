"""The split that matters: CPU on non-GEMM work while the iGPU does the GEMM.

The earlier concurrency test had both engines doing the same thing -- GEMM against GEMM -- which is
the worst possible case for contention and not how the work would actually be divided. In a real
training step the phases alternate: GEMM is 52.4% of the step, attention 18.3%, elementwise about
19%, and the optimizer the rest. Handing the GEMM to the iGPU leaves the CPU with an entirely
different kind of work.

So the question is not what the two engines achieve together on identical work. It is how much the
CPU's remaining work slows down when the iGPU is streaming memory in the background. That is the
number that decides whether an offload is worth it, and it can be measured without integrating
anything into the trainer: run real GEMM on the iGPU in a thread and time a real forward and
backward on the CPU.

The arithmetic being tested. A step is 4.5 s, of which GEMM is about 2.36 s and everything else
about 2.14 s. If the CPU keeps its 2.14 s and the iGPU handles the GEMM concurrently, the step
becomes roughly max(2.14, t_gpu) provided contention is mild. If instead the CPU's half doubles, the
2 s saved on GEMM is given back and the whole idea dies.

Three measurements, all in one session so they are comparable:

  * CPU forward+backward alone
  * iGPU GEMM alone, at its full rate
  * both at once

Decision rule, fixed in advance: the CPU's own work may not slow by more than 25%. Beyond that the
contention costs more than the offload saves at any realistic ratio of the two halves. Between 0 and
25%, the split is worth building.
"""
from __future__ import annotations

import statistics
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

import pyopencl as cl  # noqa: E402
from measure_guard import assert_idle, stamp   # noqa: E402
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

# the swizzled kernel: As padded by 1, Bs stored transposed within 4-wide blocks
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

_plat = cl.get_platforms()[0]
ctx = cl.Context(devices=[_plat.get_devices()[0]])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

assert_idle()
stamp('按任务类型分配：CPU 非 GEMM ⟷ 核显 GEMM')
print()

B, T = 2, 1024
tok = torch.load(r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt', map_location='cpu')
cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
                         ffn_dim=2048, total_tokens=T, grid=32, head_mode='flat',
                         mask_mode='causal', n_cluster=32, use_bos=False)
model = SmallARImageModelV2(cfg)
model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end, loop_L=1,
                          loop_L_max=3, loop_random=False, loop_cache='shared'))
model.train()
print('  模型: %.1fM 参数' % (sum(p.numel() for p in model.parameters()) / 1e6))

# the iGPU's GEMM workload: the model's real weight shapes, cycled continuously
SHAPES = [(2048, 512, 512), (2048, 2736, 512), (2048, 512, 2736), (2048, 2048, 512)]
rng = np.random.default_rng(7)
bufs = []
for (M, N, K) in SHAPES:
    A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
    Bm = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
    bufs.append((M, N, K,
                 cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A),
                 cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Bm),
                 cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)))
kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'gemm_fast')
FLOP = sum(2.0 * M * N * K for (M, N, K) in SHAPES)

stop = threading.Event()


def gpu_loop():
    while not stop.is_set():
        for (M, N, K, ba, bb, bc) in bufs:
            kern(queue, ((N + 63) // 64 * 16, (M + 63) // 64 * 16), (16, 16),
                 np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()


def cpu_step(n, warm=2):
    g = torch.Generator().manual_seed(0)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    ts = []
    for i in range(n + warm):
        x = tok[torch.randint(0, len(tok), (B,), generator=g)].long()
        t0 = time.perf_counter()
        out = model.forward_logits(x, None)
        lg = out[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size),
                                          x[:, 1:].reshape(-1)).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i >= warm:
            ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


torch.set_num_threads(6)
print('  --- 1. CPU 单独（6 线程）---')
cpu_alone = cpu_step(4)
print('     前向+反向: 【%.3f s】' % cpu_alone)

print('  --- 2. 核显 GEMM 单独 ---')
def gpu_loop_once():
    for (M, N, K, ba, bb, bc) in bufs:
        kern(queue, ((N + 63) // 64 * 16, (M + 63) // 64 * 16), (16, 16),
             np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    queue.finish()


gpu_t = statistics.median([(lambda t0: (gpu_loop_once(), time.perf_counter() - t0)[1])(time.perf_counter())
                           for _ in range(3)])
gpu_gf = 2.0 * FLOP / gpu_t / 1e9
print('     一组 GEMM: 【%.3f s】= %.1f GFLOPS' % (gpu_t, gpu_gf))

print('  --- 3. 同时：核显持续跑 GEMM，CPU 跑前向+反向 ---')
stop.clear()
th = threading.Thread(target=gpu_loop, daemon=True)
th.start()
time.sleep(1.5)                       # let the GPU reach steady state
cpu_contended = cpu_step(4)
stop.set()
th.join(timeout=10)
print('     CPU 前向+反向: 【%.3f s】' % cpu_contended)

print()
print('=' * 64)
print('  CPU 单独            %7.3f s' % cpu_alone)
print('  核显跑一组 GEMM      %7.3f s  (%.1f GFLOPS)' % (gpu_t, gpu_gf))
print('  争抢时 CPU          %7.3f s  ⇒ 【慢 %.1f%%】' % (cpu_contended, (cpu_contended / cpu_alone - 1) * 100))
print('=' * 64)
print()
print('  判据（事先固定）: CPU 自身工作量被拖慢 ≤ 25% ⇒ 分流值得做；> 25% ⇒ 争抢吃掉的比省下的多。')
sl = (cpu_contended / cpu_alone - 1) * 100
print('  ⇒ %s' % ('【值得做】✓' if sl <= 25 else '【不值得】✗'))
print()
print('  参考: GEMM 占一步的 52.4%，其余 47.6%')
print('  若 CPU 只剩非 GEMM 部分且拖慢 25%%，则一步约为 %.2f s 对基线 %.2f s'
      % (cpu_alone * 0.476 * 1.25, cpu_alone))
