"""Produce the 2000-step H3 verdict, ready to paste into the report's 7.31.4.

Written while the run is still going, because it is pure text processing and touches no CPU
the training needs. The moment the arm lands it can be run and the verdict written.

What it computes, all from ab_muon_results.json:

  * steps to reach a given loss, per arm, on a smoothed curve (window 50 -- the convention
    used everywhere else in this report, chosen because single-step training loss has a
    standard deviation around 0.58 and raw differences are unreadable)
  * optimiser milliseconds per step per arm
  * time-to-loss: the user's framing from 7.27.4, and the only unit that survives arms having
    different per-step costs
  * whether the early-phase deficit seen at 600 steps persists at 2000

Deliberately NOT computed: any end-to-end speedup figure. The harness's forward/backward is
not decomposed, so converting an optimiser saving into a whole-step saving would need a
measurement that has not been made, and the plan forbids inferring it.

Uses no numpy and no torch: json, statistics and arithmetic only.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import io
import json
import os
import statistics
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

ROOT = r'D:\work\bitsandbytes-CPU'
J = os.path.join(ROOT, 'ab_muon_results.json')

j = json.load(io.open(J, encoding='utf-8'))
curves = {}
for name in ('adamw', 'muon', 'muon3'):
    v = j.get(name)
    if isinstance(v, list) and v:
        curves[name] = [float(x if isinstance(x, (int, float)) else x[-1]) for x in v]
    elif isinstance(v, dict):
        c = v.get('losses') or v.get('loss') or v.get('curve')
        if c:
            curves[name] = [float(x if isinstance(x, (int, float)) else x[-1]) for x in c]

print('  已落盘的臂: %s' % ', '.join('%s(%d 点)' % (k, len(v)) for k, v in curves.items()))
if 'muon3' not in curves:
    print()
    print('  muon3 还没落盘 —— 2000 步仍在跑。此脚本等它出现即可直接重跑。')
    sys.exit(0)


def smooth(v, w=50):
    out, s = [], 0.0
    for i, x in enumerate(v):
        s += x
        if i >= w:
            s -= v[i - w]
        out.append(s / min(i + 1, w))
    return out


def first_at(c, t, lo=0):
    for i in range(lo, len(c)):
        if c[i] <= t:
            return i + 1
    return None


sm = {k: smooth(v) for k, v in curves.items()}

lines = []
def W(s=''):
    lines.append(s)
    print(s)

W('  步数     AdamW     Muon(5)   Muon3(3)   3 与 5 之差')
for s in (50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1800, 2000):
    if s > len(curves['muon3']):
        break
    row = ['  %-6d' % s]
    for k in ('adamw', 'muon', 'muon3'):
        row.append(' %9.4f' % curves[k][s - 1] if s <= len(curves[k]) else '         —')
    d = curves['muon3'][s - 1] - curves['muon'][s - 1] if s <= len(curves['muon']) else None
    row.append('   %+.4f' % d if d is not None else '')
    W(''.join(row))

W()
W('  到同一 loss 的步数（平滑窗口 50）:')
W('    %-10s %10s %10s %10s' % ('目标', 'NS=5', 'NS=3', '差'))
targets = [7.0, 6.8, 6.6, 6.4, 6.2, 6.0, 5.9, 5.8, 5.7, 5.6, 5.5]
early = {}
for t in targets:
    n5 = first_at(sm['muon'], t)
    n3 = first_at(sm['muon3'], t)
    if n5 and n3:
        early[t] = (n5, n3)
        W('    %-10.1f %10d %10d %+9.1f%%' % (t, n5, n3, (n3 / n5 - 1) * 100))
    else:
        W('    %-10.1f %10s %10s' % (t, n5 or '未达到', n3 or '未达到'))

W()
W('  末值: %s' % ' · '.join('%s %.4f' % (k, curves[k][-1]) for k in curves))

# optimiser cost
W()
W('  优化器耗时（同一 harness，实测）:')
opts = {}
for k in curves:
    v = j.get(k)
    if isinstance(v, dict):
        for key in ('opt_ms_per_step', 'opt_ms', 'optimizer_ms_per_step'):
            if key in v:
                opts[k] = float(v[key])
                break
for k, v in opts.items():
    W('    %-6s %8.1f ms/步' % (k, v))
if 'muon3' in opts and 'muon' in opts:
    W('    ⇒ NS=5→3 省 %.1f ms/步（%.1f%%）'
      % (opts['muon'] - opts['muon3'], (1 - opts['muon3'] / opts['muon']) * 100))

W()
W('  判据对照:')
if 5.5 in early:
    n5, n3 = early[5.5]
    W('    loss ≤ 5.5 ⇒ NS=5 %d 步 · NS=3 %d 步 ⇒ 差 %+.1f%%' % (n5, n3, (n3 / n5 - 1) * 100))
if 6.0 in early:
    n5, n3 = early[6.0]
    W('    loss ≤ 6.0 ⇒ NS=5 %d 步 · NS=3 %d 步 ⇒ 差 %+.1f%%（600 步时为 +12%%）'
      % (n5, n3, (n3 / n5 - 1) * 100))
    W('    ⇒ 若该差值随训练长度收窄 ⇒ 早期落后是暂态；若不收窄 ⇒ 是真实差距')

W()
W('  不做的事: 不换算整步加速比。本 harness 的前向/反向未分解，')
W('  把优化器节省换算到整步需要一次未做的测量，计划明令禁止推断。')

io.open(os.path.join(ROOT, 'ns_h3_2000_verdict.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print()
print('  已写出 ns_h3_2000_verdict.txt')
