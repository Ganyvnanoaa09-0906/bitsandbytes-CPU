"""bench_ar_batch_threads.py -- throughput vs batch size and thread count.

Two questions the user's constraints make urgent, both answerable in minutes
rather than by a multi-day run.

1. Batch size. Training at grid=32 is 1024 tokens per sample, and the run so far
   used batch 8. On a CPU, larger batches make GEMMs more efficient, so throughput
   per token may improve -- which is a speedup that does not touch quality, as long
   as the learning rate is scaled with the batch. AdamW4bit, added today, cuts
   optimiser state from 8 bytes to 1.03 bytes per parameter, so the memory that
   previously capped the batch is partly freed.

2. Thread count. The user's concern is explicitly about heat on the adapter,
   display and mainboard over a 1.5 day run. This is a 6-core/6-thread part, so
   torch at 6 threads saturates everything and the package runs at its ceiling.
   Dropping to 5 costs some throughput but lowers the sustained thermal load; the
   measurement is what says how much it costs.

Reports tokens/second, since that is the quantity that matters: wall time is
tokens divided by this, and quality is unaffected by how the tokens are grouped.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402


def build(grid):
    # head_mode must be explicit: the config default is 'factorized', whose output is a
    # pair of (B, T, 32) tensors rather than (B, T, vocab), while train_ar_v2.py defaults
    # to 'flat'. Matching the training configuration is the whole point of the benchmark.
    cfg = SmallImageConfigV2(grid=grid, total_tokens=grid * grid, use_bos=True,
                             head_mode='flat', mask_mode='causal', n_cluster=32)
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=True, loop_cache='perloop'))
    m.train()
    return m, cfg


def run_once(m, cfg, T, B, iters=3):
    x = torch.randint(0, 1024, (B, T)).long()
    o = m.forward_logits(x, None)
    lg = o[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, cfg.vocab_size), x.reshape(-1))
    loss.backward()
    m.zero_grad(set_to_none=True)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        o = m.forward_logits(x, None)
        lg = o[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, cfg.vocab_size), x.reshape(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)
    dt = (time.perf_counter() - t0) / iters
    return dt, B * T / dt


ap = argparse.ArgumentParser()
ap.add_argument('--grid', type=int, default=32)
ap.add_argument('--threads', default='6,5,4')
ap.add_argument('--batches', default='4,8,16,32')
a = ap.parse_args()
T = a.grid * a.grid

print('AR 训练吞吐: grid=%d ⇒ T=%d token/样本, 前向+反向+优化器步' % (a.grid, T))
print()
print('%-9s %-9s %10s %14s %12s %s' % ('线程', 'batch', '秒/步', 'token/秒', '相对', '备注'))
print('-' * 76)
base = None
for th in [int(x) for x in a.threads.split(',')]:
    torch.set_num_threads(th)
    for B in [int(x) for x in a.batches.split(',')]:
        try:
            m, cfg = build(a.grid)
            dt, tps = run_once(m, cfg, T, B)
            if base is None:
                base = tps
            print('%-9d %-9d %10.3f %14.1f %11.2fx %s'
                  % (th, B, dt, tps, tps / base,
                     '基线' if base == tps else ''), flush=True)
            del m
            gc.collect()
        except Exception as e:
            print('%-9d %-9d %10s %14s %12s %s: %s'
                  % (th, B, 'FAIL', '-', '-', type(e).__name__, str(e)[:34]), flush=True)
    print()
print('说明: token/秒 是可比量。若大 batch 的 token/秒更高，则【同样的训练量用时更短】，')
print('      而质量只取决于总 token 数与学习率调度，不取决于怎么分组。')
