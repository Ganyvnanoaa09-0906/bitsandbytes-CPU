# -*- coding: utf-8 -*-
"""test_loop.py — 验证「第4~7层跑两遍」到底烧掉多少算力。

run_blocks 里：
    if i in loop_idx:
        for _ in range(loop_times - 1):
            x = blk(x, ...)          # 再跑一遍
所以 loop_idx=[4,5,6,7] + loop_times=2 时，10 层的模型实际跑 14 个 block pass。

这个循环是【权重共享】的，所以参数没变多 —— 纯粹是 FLOPs 浪费（如果它没用的话）。
"""
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

T, B = 256, 8


def build(loop_times):
    cfg = SmallImageConfigV2(
        vocab_size=1024, total_tokens=T, grid=16,
        d_model=512, n_layer=10, n_head=8, n_kv_head=4,
        ffn_dim=2048, head_mode='flat', cond_mode='none',
        pos_mode='2d', token_order='raster',
        loop_start=4, loop_end=7, loop_times=loop_times,
    )
    return SmallARImageModelV2(cfg)


def measure(loop_times, threads=6, iters=4):
    torch.set_num_threads(threads)
    m = build(loop_times)
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x = torch.randint(0, 1024, (B, T))
    y = torch.randint(0, 1024, (B, T))

    def step():
        out = m.forward_logits(x)
        gen = out[1]
        logits = gen[0] if isinstance(gen, tuple) else gen
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(2):
        step()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    dt = (time.perf_counter() - t0) / iters
    del m, opt, x, y
    return dt, nparam


print('=' * 66, flush=True)
print('loop_times 消融（b=%d T=%d threads=6）' % (B, T), flush=True)
print('=' * 66, flush=True)

res = {}
for lt in (1, 2):
    dt, nparam = measure(lt)
    res[lt] = dt
    passes = 10 + (4 if lt >= 2 else 0)
    print('  loop_times=%d   %d 个 block pass   %6.3f s/步   %7.1f tok/s   %.2fM 参数'
          % (lt, passes, dt, B * T / dt, nparam / 1e6), flush=True)

if 1 in res and 2 in res:
    print()
    print('  关掉循环 = 快 %.1f%%   (%.3f -> %.3f s/步)'
          % (100 * (res[2] - res[1]) / res[2], res[2], res[1]), flush=True)

print()
print('DONE', flush=True)
