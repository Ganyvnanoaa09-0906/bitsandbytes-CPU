"""profile_ar_step.py -- what actually consumes the AR training step?

GDN replaces O(T^2) attention with an O(T) recurrence and measures 5-15x faster as
an operator, yet end-to-end it changes nothing: 0.78x at T=256, 0.79x at 576, 0.99x
at 1024. So attention is not the bottleneck here, and the inference I drew earlier
-- that 4x the tokens costing 6.41x the time implied attention dominance -- was
wrong, because that ratio mixed in batch and loop-count changes. Same failure as
the rest of today: a criterion that did not measure what it was being used to
conclude.

The question that matters is therefore what does consume the step. Answer it with
torch's profiler rather than by reasoning: run one forward+backward and print the
operators by self time, then the same for the components the model is built from.
Guessing has been expensive today.

The model is built with head_mode='flat' to match train_ar_v2.py's default; the
config default is 'factorized', which changes the output head's shape entirely.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

torch.set_num_threads(6)
GRID = 32
B = 2

cfg = SmallImageConfigV2(grid=GRID, total_tokens=GRID * GRID, use_bos=True,
                         head_mode='flat', mask_mode='causal')
m = SmallARImageModelV2(cfg)
m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                      loop_L=1, loop_L_max=3, loop_random=True, loop_cache='perloop'))
m.train()
T = cfg.total_tokens
x = torch.randint(0, cfg.vocab_size, (B, T)).long()
print('grid=%d ⇒ T=%d, batch=%d, d_model=%d, n_layer=%d, loop L=1..3'
      % (GRID, T, B, cfg.d_model, cfg.n_layer))


def step():
    o = m.forward_logits(x, None)
    lg = o[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, cfg.vocab_size), x.reshape(-1))
    loss.backward()
    m.zero_grad(set_to_none=True)
    return loss


step()
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(3):
        step()

print()
print('=== 按自身耗时排序的算子（前 18）===')
print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=18))
