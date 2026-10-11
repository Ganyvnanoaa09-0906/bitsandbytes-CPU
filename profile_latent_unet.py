# -*- coding: utf-8 -*-
"""profile_latent_unet.py — 量【16x16 潜变量】UNet 的真实算子占比，决定该手搓哪个内核。

为什么必须量（2026-09-19）：
  容器实验给出的结论是"GN+SiLU 只占 ResBlock 的 0.5~1.5%，卷积是大头"，
  但那是 **SD 的 UNet（64x64x320）** 上的数字。
  我们的扩散跑在 **16x16x64** 上 —— 空间小 16 倍、通道小 5 倍。
  卷积的 FLOPs 随 H*W*C_in*C_out 走，而 norm/激活/元素级算子只随 H*W*C 走；
  空间一缩，卷积的"单位收益"下降得比元素级算子快，
  ⇒ **算子占比会完全换位**。直接照搬 SD 的结论会把手搓算力放错地方。

本脚本用 torch.profiler 给出按算子类别的耗时占比（分别看 forward / 训练步）。
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def classify(name):
    n = name.lower()
    if 'conv' in n:
        return 'conv (卷积)'
    if 'groupnorm' in n or 'batchnorm' in n or 'layernorm' in n or 'norm' in n:
        return 'norm (归一化)'
    if 'silu' in n or 'gelu' in n or 'relu' in n or 'sigmoid' in n or 'mish' in n:
        return 'act (激活)'
    if 'attention' in n or 'scaled_dot' in n:
        return 'attention (注意力)'
    if 'linear' in n or 'addmm' in n or 'matmul' in n or 'mm' in n:
        return 'linear/GEMM'
    if 'mul' in n or 'add' in n or 'elementwise' in n or 'cat' in n or 'upsample' in n:
        return 'elementwise/其他'
    return '其他'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=16)
    ap.add_argument('--ch', default='128,256,256')
    ap.add_argument('--in-ch', type=int, default=64)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--grad', action='store_true', help='同时量训练步（含反向）')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from train_latent_diffusion import build_unet
    ch = [int(x) for x in a.ch.split(',')]
    model = build_unet(ch, sample_size=a.size, in_ch=a.in_ch)
    npar = sum(p.numel() for p in model.parameters())
    print('[prof] UNet sample=%d in_ch=%d ch=%s  params=%.2fM  batch=%d  threads=%d'
          % (a.size, a.in_ch, ch, npar / 1e6, a.batch, a.threads))

    x = torch.randn(a.batch, a.in_ch, a.size, a.size)
    t = torch.randint(0, 1000, (a.batch,))

    def run_fwd():
        with torch.no_grad():
            return model(x, t).sample

    # warmup（CPU 必须预热，否则第一次测量偏低）
    for _ in range(3):
        run_fwd()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        run_fwd()
    fwd_ms = (time.perf_counter() - t0) / a.iters * 1000
    print('[prof] forward 单次 %.2f ms' % fwd_ms)

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(a.iters):
            run_fwd()
    agg = {}
    for e in prof.key_averages():
        k = classify(e.key)
        agg[k] = agg.get(k, 0.0) + e.self_cpu_time_total
    tot = sum(agg.values()) or 1.0
    print()
    print('=== forward 算子占比（self CPU time）===')
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print('  %-22s %8.2f ms  %6.1f%%' % (k, v / 1000, 100 * v / tot))
    print('  %-22s %8.2f ms' % ('TOTAL', tot / 1000))

    if a.grad:
        model.train()
        for _ in range(2):
            model(x, t).sample.sum().backward()
            model.zero_grad()
        t0 = time.perf_counter()
        for _ in range(max(3, a.iters // 4)):
            model(x, t).sample.sum().backward()
            model.zero_grad()
        n = max(3, a.iters // 4)
        tr_ms = (time.perf_counter() - t0) / n * 1000
        print()
        print('[prof] 训练步单次 %.2f ms (fwd+bwd，约 forward 的 %.1fx)'
              % (tr_ms, tr_ms / max(fwd_ms, 1e-9)))
        with profile(activities=[ProfilerActivity.CPU]) as prof2:
            for _ in range(n):
                model(x, t).sample.sum().backward()
                model.zero_grad()
        agg2 = {}
        for e in prof2.key_averages():
            k = classify(e.key)
            agg2[k] = agg2.get(k, 0.0) + e.self_cpu_time_total
        tot2 = sum(agg2.values()) or 1.0
        print('=== 训练步算子占比 ===')
        for k, v in sorted(agg2.items(), key=lambda kv: -kv[1]):
            print('  %-22s %8.2f ms  %6.1f%%' % (k, v / 1000, 100 * v / tot2))

    print()
    print('⇒ 占比最高的那 1~2 类才是值得手搓内核的地方。')


if __name__ == '__main__':
    main()
