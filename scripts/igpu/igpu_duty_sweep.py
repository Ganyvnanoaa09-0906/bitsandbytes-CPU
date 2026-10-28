"""Resolve the contradiction: the CPU slowdown as a function of the iGPU's duty cycle.

Two measurements disagreed twentyfold, and both look sound. The difference between them is the
iGPU's intensity, which neither recorded.

  task split    one group of GEMMs, then queue.finish() before the next -- the GPU idles between
                groups. 506.9 GFLOPS, CPU slows 4.3%.
  GIL retest    a tight loop for 25 seconds with no gap. 789.5 GFLOPS, CPU slows 94.4%.

Same kernel, same shapes, same machine. The second is 56% faster on the GPU side, which means the
first was leaving the device idle part of the time, and it is therefore not surprising that it
interfered less. The hypothesis is that the CPU slowdown is a function of how hard the iGPU is
actually working, and that both numbers are points on one curve.

That matters because it decides the practical question. A real offload hands the GEMM portion --
52.4% of a step -- to the iGPU, and during that portion the device runs flat out, which is the
intensive regime, not the bursty one. So if the curve is steep, the offload is not viable however
good the low-duty numbers look.

Sweep the duty cycle from 0 to 100% by sleeping between groups, and measure the CPU step time at
each point. Also record the iGPU's own rate at each duty, so that a slowdown can be attributed to
work rather than to clock throttling.

Decision rule, fixed in advance: if the CPU slows by more than 25% at 75% duty or above, the offload
is not viable and 7.46's closure stands on better evidence. If the curve stays flat, then the
intensive measurement was the anomaly and the offload is worth building.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

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
        { int f4 = lane; int r = (f4*4)/TK, c = (f4*4)%TK;
          int gr = by + r, gc = k0 + c; float4 v = (float4)0.f;
          if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr*K + gc);
          As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w; }
        { int f4 = lane; int k = (f4*4)/TS, n = (f4*4)%TS;
          int gk = k0 + k, gn = bx + n; float4 v = (float4)0.f;
          if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk*N + gn);
          #pragma unroll
          for (int q = 0; q < 4; q++) { int cc = n + q;
              Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                  (q==0)?v.x:((q==1)?v.y:((q==2)?v.z:v.w)); } }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty*TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j*(TS/TN) + tx];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i]*b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) { int gr = by + ty*TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) { int gc = bx + tx*TN + j;
            if (gr < M && gc < N) C[gr*N + gc] = acc[i][j]; } }
}
'''

SHAPES = [(2048, 512, 512), (2048, 2736, 512), (2048, 512, 2736), (2048, 2048, 512)]
FLOP = sum(2.0 * M * N * K for (M, N, K) in SHAPES)
DUTY = [0.0, 0.25, 0.5, 0.75, 1.0]


def gpu_worker(duty, seconds, q):
    import pyopencl as cl
    plat = cl.get_platforms()[0]
    ctx = cl.Context(devices=[plat.get_devices()[0]])
    cq = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    rng = np.random.default_rng(11)
    bufs = []
    for (M, N, K) in SHAPES:
        A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
        Bm = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
        bufs.append((M, N, K, cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A),
                     cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Bm),
                     cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)))
    kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'gemm_fast')

    def grp():
        for (M, N, K, ba, bb, bc) in bufs:
            kern(cq, ((N + 63)//64*16, (M + 63)//64*16), (16, 16),
                 np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        cq.finish()

    grp()                       # warm
    t0 = time.perf_counter()
    on = 0.0
    n = 0
    while time.perf_counter() - t0 < seconds:
        ts = time.perf_counter()
        grp()
        on += time.perf_counter() - ts
        n += 1
        if duty < 1.0:
            _el = time.perf_counter() - t0
            # cumulative off-time so that on/(on+off) equals the target duty
            time.sleep(max(0.0, on / max(duty, 1e-6) - _el))
    el = time.perf_counter() - t0
    q.put((n, on, el, 2.0 * FLOP * n / max(on, 1e-9) / 1e9))


def main():
    from scripts.lib.measure_guard import assert_idle, stamp
    from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    from scripts.lib.loop_transformer import LoopConfig

    assert_idle()
    stamp('核显强度扫描：CPU 拖慢 vs 占空比')
    torch.set_num_threads(6)
    tok = torch.load(r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt', map_location='cpu')
    B, T = 2, 1024
    cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
                             ffn_dim=2048, total_tokens=T, grid=32, head_mode='flat',
                             mask_mode='causal', n_cluster=32, use_bos=False)
    model = SmallARImageModelV2(cfg)
    model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end, loop_L=1,
                              loop_L_max=3, loop_random=False, loop_cache='shared'))
    model.train()

    def cpu_once():
        g = torch.Generator().manual_seed(0)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        ts = []
        for i in range(6):
            x = tok[torch.randint(0, len(tok), (B,), generator=g)].long()
            t0 = time.perf_counter()
            out = model.forward_logits(x, None)
            lg = out[1]
            lg = lg[0] if isinstance(lg, tuple) else lg
            torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size),
                                              x[:, 1:].reshape(-1)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts[2:])

    print()
    print('  %-8s %12s %12s %10s' % ('核显占空比', 'CPU s/步', '相对基线', '核显 GFLOPS'))
    print('  ' + '-' * 48)
    base = None
    for d in DUTY:
        if d == 0.0:
            t = cpu_once()
            base = t
            print('  %-8s %12.3f %11s %10s' % ('0%', t, '1.000×', '—'))
            continue
        q = mp.Queue()
        p = mp.Process(target=gpu_worker, args=(d, 12.0, q))
        p.start()
        time.sleep(2.5)
        t = cpu_once()
        n, on, el, gf = q.get()
        p.join()
        print('  %-8s %12.3f %10.3f× %10.1f'
              % ('%d%%' % int(d * 100), t, t / base, gf))
    print()
    print('  判据（事先固定）: 占空比 ≥75% 时 CPU 拖慢 > 25% ⇒ 卸载不可行,7.46 的关闭成立；')
    print('                    若曲线平坦 ⇒ 满载那次是异常,卸载值得做。')
    print()
    print('  说明: 占空比用「跑一组然后 sleep」实现;核显 GFLOPS 按实际在跑的时间算。')


if __name__ == '__main__':
    mp.freeze_support()
    main()
