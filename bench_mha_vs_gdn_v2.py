"""bench_mha_vs_gdn_v2.py -- does swapping attention for GDN actually speed up this model?

video_attn_vs_gdn.py compared the two operators in isolation and the ratio grows
with T, which says linearisation is worthwhile in principle. This checks it inside
the real model, where attention is only one of several components: if the rest of
the block dominates, the end-to-end gain will be much smaller than the operator
ratio, and the 22-hour estimate would barely move.

Runs the same model twice, differing only in cfg.attn_kind, at T = 256, 576 and
1024, reporting seconds per step and tokens per second. Loss is printed too so it
is visible that both are actually computing a language-modelling objective; this is
a speed comparison, not a quality one -- GDN is a different architecture and its
quality has to be established by training, which is the next step, not this one.

head_mode is set to 'flat' explicitly because the config default is 'factorized',
whose forward returns a pair of (B,T,n_cluster) tensors instead of (B,T,vocab).
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

torch.set_num_threads(6)


def build(kind, grid, chunk=64):
    cfg = SmallImageConfigV2(grid=grid, total_tokens=grid * grid, use_bos=True,
                             head_mode='flat', mask_mode='causal')
    cfg.attn_kind = kind
    cfg.chunk_size = chunk
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=True, loop_cache='perloop'))
    m.train()
    return m, cfg


def timed(m, cfg, B, T, reps=3):
    x = torch.randint(0, cfg.vocab_size, (B, T)).long()

    def step():
        o = m.forward_logits(x, None)
        lg = o[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, cfg.vocab_size), x.reshape(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)
        return float(loss.detach())

    l0 = step()
    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    dt = (time.perf_counter() - t0) / reps
    return dt, B * T / dt, l0


print('AR 模型端到端: mha vs gdn（只改 attn_kind ✓ 其余完全相同）')
print()
print('%-6s %-7s %10s %12s %12s %8s' % ('T', 'attn', 's/步', 'token/秒', '相对 mha', 'loss'))
print('-' * 62)
for T in (256, 576, 1024):
    grid = int(T ** 0.5)
    if grid * grid != T:
        continue
    base = None
    for kind in ('mha', 'gdn'):
        try:
            m, cfg = build(kind, grid)
            n = sum(p.numel() for p in m.parameters())
            dt, tps, l0 = timed(m, cfg, 2, T)
            if kind == 'mha':
                base = tps
            print('%-6d %-7s %10.3f %12.1f %11.2fx %8.4f'
                  % (T, kind, dt, tps, tps / base, l0))
            if kind == 'gdn':
                print('%-6s %-7s %10s %12s %12s %8s'
                      % ('', '(参数 %.1fM)' % (n / 1e6), '', '', '', ''))
            del m
        except Exception as e:
            print('%-6d %-7s %10s %12s %12s %8s' % (T, kind, 'FAIL', '-', '-', str(e)[:26]))
    print()
print('读法: 相对 mha > 1 表示 GDN 更快。若接近 1，说明注意力不是该形状下的瓶颈，')
print('      换架构对总时长帮助有限；video_attn_vs_gdn 的 5~15x 是纯算子倍率。')
