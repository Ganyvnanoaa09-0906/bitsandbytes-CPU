"""Shape engineering: the one speed lever that does not need new hardware.

The user's i5 is only available during holidays, so "use the other machine" is
not an option -- software optimisation is the whole game. And today's own shape
sweep contains an unexploited signal:

    2048x512x512   174.5 GFLOPS
    2048x512x2048  247.7 GFLOPS     <- +42% just from a wider N
    2048x4096x512  182.2
    512x512x512    211.2

The model splits qkv into three separate N=512 GEMMs (fuse_qkv=False in the
config), and SwiGLU into two N=ffn GEMMs. If width is what pays, fusing should
recover part of that 42% -- with no change to the maths.

This measures, on this machine, in one run:
  A) qkv as 3 x (T,512)x(512,512)   vs   1 x (T,512)x(512,1536)
  B) gate/up as 2 x (T,512)x(512,2048) vs 1 x (T,512)x(512,4096)
  C) the mlp down and head as-is, for reference
and reports fwd and fwd+bwd separately (training cares about the sum).
"""
import sys
import time

import torch

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(6)
T, B, D, F, V = 256, 8, 512, 2048, 1024
TOK = T * B
print('threads=%d  tokens/step=%d\n' % (torch.get_num_threads(), TOK))


def bench(name, mats, reps=3, backward=False):
    """mats: list of (in, out) -- each becomes A[TOK,in] @ W[in,out],
    optionally with a backward pass so the training cost is represented."""
    As = [torch.randn(TOK, i, requires_grad=backward) for (i, _o) in mats]
    Ws = [torch.randn(i, o, requires_grad=backward) for (i, o) in mats]
    fl = sum(2.0 * TOK * i * o for (i, o) in mats) * (3 if backward else 1)
    def once():
        ys = [a @ w for a, w in zip(As, Ws)]
        if backward:
            s = sum(y.float().pow(2).mean() for y in ys)
            s.backward()
            for t in list(As) + list(Ws):
                if t.grad is not None:
                    t.grad = None
        return ys
    once()                                   # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        once()
    dt = (time.perf_counter() - t0) / reps
    g = fl / dt / 1e9
    print('  %-34s %8.1f ms %9.1f GFLOPS' % (name, dt * 1000, g))
    return g, dt


print('=== 前向 ===')
bench('qkv 拆三个 (N=512 x3)', [(D, D), (D, D), (D, D)])
bench('qkv 合一 (N=1536)', [(D, 3 * D)])
bench('mlp gate/up 拆两个 (N=2048 x2)', [(D, F), (D, F)])
bench('mlp gate/up 合一 (N=4096)', [(D, 2 * F)])
bench('mlp down (F->D)', [(F, D)])
bench('head (D->V)', [(D, V)])

print('\n=== 前向+反向（训练实际关心的）===')
# note: to keep shapes honest, model the three qkv slices as one wide matmul's
# gradient too; backward cost scales with the same FLOPs either way.
bench('qkv 拆三个', [(D, D), (D, D), (D, D)], backward=True)
bench('qkv 合一', [(D, 3 * D)], backward=True)
bench('gate/up 拆两个', [(D, F), (D, F)], backward=True)
bench('gate/up 合一', [(D, 2 * F)], backward=True)

print('\n判读：')
print('  若"合一"明显更快 ⇒ 把 fuse_qkv 打开（配置里现成的开关）就能白拿一部分')
print('  若两者相同 ⇒ 宽度不是这条曲线的原因，别改结构，去别处找')
