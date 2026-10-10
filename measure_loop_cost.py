"""measure_loop_cost.py -- what does the loop transformer cost in the current config?

SPEED_FINDINGS.md, written 2026-09-16, already contains the answer to the question
that consumed last night: it ablated the loop block and measured 3.452 s/step with
loop_times=2 against 2.505 with loop_times=1 -- 38% faster with the parameter count
unchanged, because the loop re-runs blocks 4..7 with the same weights, so it saves
parameters rather than FLOPs, and this workload is compute-bound. It also notes that
train_ar_v2.py never passes those two arguments, so the cost was being paid by
default rather than by choice, and cites ELT (arXiv 2604.09168) for the finding that
naive looping only pays off with intra-loop self-distillation, which this model does
not have.

The AR model has a second, related knob: LoopConfig with loop_L/loop_L_max, which is
what my own benchmarks used (loop_L=1, loop_L_max=3, loop_random=True). This measures
its cost directly, at the shapes actually in use, so the two mechanisms can be told
apart and the cheaper setting identified without changing what the model computes
beyond the loop count.

Configs swept: loop_L_max in {1,2,3}, loop_random on and off.
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
GRID = 32
B = 2


def build(L, Lmax, random, cache='perloop'):
    cfg = SmallImageConfigV2(grid=GRID, total_tokens=GRID * GRID, use_bos=True,
                             head_mode='flat', mask_mode='causal')
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=L, loop_L_max=Lmax, loop_random=random,
                          loop_cache=cache))
    m.train()
    return m, cfg


def timed(m, cfg, reps=3):
    T = cfg.total_tokens
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
    return (time.perf_counter() - t0) / reps, B * T / ((time.perf_counter() - t0) / reps), l0


print('grid=%d ⇒ T=%d, batch=%d, 线程 6' % (GRID, GRID * GRID, B))
print()
print('%-28s %10s %12s %12s %8s' % ('配置', 's/步', 'token/秒', '相对最快', 'loss'))
print('-' * 76)
rows = []
# ★ 关键是最后两行: 真训练【不传 loop_cache】（默认 shared）也【不传 loop_random】（默认 False）
for (L, Lmax, rnd, lc, label) in [
    (1, 3, True,  'perloop', '我昨晚用的（random + perloop）'),
    (1, 3, False, 'perloop', '固定 3 + perloop'),
    (1, 3, False, 'shared',  '★ 真训练的默认（shared，无 random）'),
    (1, 1, False, 'shared',  'loop_L=1 + shared'),
    (1, 3, True,  'shared',  'random + shared'),
    (3, 3, False, 'shared',  '固定 L=3 + shared'),
]:
    try:
        m, cfg = build(L, Lmax, rnd, lc)
        dt, tps, l0 = timed(m, cfg)
        rows.append((tps, label, dt, l0))
        print('%-32s %10.3f %12.1f %12s %8.4f' % (label, dt, tps, '', l0))
        del m
    except Exception as e:
        print('%-32s %10s %12s %12s %8s' % (label, 'FAIL', '-', '-', str(e)[:24]))
print()
if rows:
    best = max(rows)
    print('  ⇒ 最快: %s  %.1f token/秒' % (best[1], best[0]))
    for tps, label, dt, l0 in sorted(rows, reverse=True):
        print('     %-28s %.2fx 于最慢' % (label, tps / min(r[0] for r in rows)))
    print()
    print('  对照 SPEED_FINDINGS.md: loop_times 2→1 是 +38%（593→818 tok/s，参数量不变）')
    print('  这里测的是另一个旋钮（LoopConfig 的 loop_L/L_max），两者都要付算力。')
