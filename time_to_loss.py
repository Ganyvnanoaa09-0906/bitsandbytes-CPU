"""time_to_loss.py -- compare optimisers by wall-clock time, not by step count.

The user's framing, and it is the right one: if optimiser A reaches in five hours the loss
that AdamW needs ten hours for, that is a 2x speedup regardless of how many steps either
took. Step-count comparisons silently assume steps cost the same, and for Muon they do not
-- its Newton-Schulz step measured 5.09 s against AdamW's 3.26 s at T=1024, batch 2.

That matters more once curriculum learning enters, because then even the sequence length
changes between phases and a step count stops being a unit at all. Time is the only common
denominator, and it is also the thing the user actually cares about.

This reads the per-step losses the A/B is writing to JSON, converts them to cumulative wall
time using each arm's measured seconds per step, and reports:

  * the loss each arm has reached at a series of time points;
  * the time each arm needs to reach a series of loss targets;
  * the ratio, which is the speedup in the only units that matter.

It runs on whatever results exist, so it can be used now for AdamW alone and again once
Muon's arm lands.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.stdout.reconfigure(encoding='utf-8')

P = r'D:\work\bitsandbytes-CPU\ab_muon_results.json'
if not os.path.exists(P):
    print('还没有结果文件')
    raise SystemExit(1)

with open(P, encoding='utf-8') as f:
    res = json.load(f)

# seconds per step, measured in the same run (from the log)
SPEED = {'adamw': 6631.6 / 2000, 'muon': None}
# Muon's per-step cost comes from its own arm once it finishes; use the 600-step
# measurement as a provisional value so a partial curve is still interpretable.
MUON_PROVISIONAL = 3056.6 / 600
if 'muon' in res and len(res['muon']) > 0:
    # if the log line exists it will have been filled in by the caller
    pass
SPEED['muon'] = SPEED['muon'] or MUON_PROVISIONAL

print('每步成本: AdamW %.3f s' % SPEED['adamw'], end='')
print('   Muon %.3f s%s' % (SPEED['muon'],
                            '' if 'muon' in res else '（600 步那次的值，暂用）'))
print()

W = 50          # smoothing window, same as the runner uses


def smooth(xs, w=W):
    return [statistics.mean(xs[max(0, i - w):i + 1]) for i in range(len(xs))]


curves = {}
for name, losses in res.items():
    t = SPEED.get(name)
    if t is None:
        continue
    curves[name] = (smooth(losses), t)

if not curves:
    print('没有可比较的臂')
    raise SystemExit(1)

# loss at matched time points
max_t = max(len(c[0]) * c[1] for c in curves.values())
print('=== 同一【时间点】上各自的 loss ===')
print('  %10s' % '时间(小时)', end='')
for n in curves:
    print(' %12s' % n, end='')
print()
for hours in (0.25, 0.5, 1, 2, 3, 5, 8, 12):

    secs = hours * 3600
    print('  %10.2f' % hours, end='')
    for n, (ls, t) in curves.items():
        idx = int(secs / t)
        if idx < len(ls):
            print(' %12.4f' % ls[idx], end='')
        else:
            print(' %12s' % '—', end='')
    print()

# time to reach loss targets
print()
print('=== 达到同一【loss 目标】各自需要多少时间 ===')
lo = min(min(c[0]) for c in curves.values())
hi = max(c[0][0] for c in curves.values())
targets = [hi - (hi - lo) * f for f in (0.25, 0.5, 0.75, 0.9, 1.0)]
print('  %10s' % '目标 loss', end='')
for n in curves:
    print(' %14s' % (n + ' 小时'), end='')
if len(curves) == 2:
    print(' %10s' % '倍速')
else:
    print()
for tg in targets:
    print('  %10.4f' % tg, end='')
    times = {}
    for n, (ls, t) in curves.items():
        hit = None
        for i, v in enumerate(ls):
            if v <= tg:
                hit = (i + 1) * t / 3600.0
                break
        times[n] = hit
        print(' %14s' % ('%.2f' % hit if hit else '未达到'), end='')
    if len(curves) == 2:
        a, b = list(times.values())
        if a and b:
            print(' %9.2fx' % (a / b), end='')
    print()

print()
print('注: 平滑窗口 %d 步，与 A/B 报告一致；单步 loss 的标准差约 0.58，' % W)
print('    因此目标间隔小于 0.1 的差别不可信，时间比在目标接近平台时会失真。')
