"""Compare the trainer's own Muon+NS=3 run against the AdamW baseline, both on the same path.

Every previous speed figure for Muon borrowed its step-count ratio from ab_muon_v2.py. This
compares two runs that both went through train_ar_v2.py under identical settings:

    A   --optimizer adamw                  i5build/sgdrA.log (already recorded)
    B   --optimizer muon --ns-iters 3      trainer_muon3.log (running now)

Both are 1500 steps, batch 2, T=1024, validation every 250 steps on the same held-out set.

Reported, with the conventions used everywhere else in this report:

  * validation loss at each checkpoint, both arms, side by side
  * steps to reach a given validation loss, on a smoothed curve
  * wall clock to reach it, which is the user's framing from 7.27.4 and the only unit that
    survives the arms having different per-step costs
  * the step-count ratio and the step-cost ratio separately, then their product, so each
    component can be checked rather than only the conclusion

Deliberately not done: any claim about generation quality. Validation loss only, as in every
other comparison here, and that limitation is stated in the output.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import io
import os
import re
import statistics
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

ROOT = r'D:\work\bitsandbytes-CPU'


def curve(path):
    """(step, val) pairs and per-step times from a train_ar_v2.py log."""
    vals, steps = [], []
    try:
        for ln in io.open(path, encoding='utf-8', errors='replace'):
            m = re.search(r'step (\d+)\s+train=[\d.]+\s+val=([\d.]+)', ln)
            if m:
                vals.append((int(m.group(1)), float(m.group(2))))
            m2 = re.search(r'(\d+)/1500 loss=[\d.]+ ([\d.]+)s', ln)
            if m2:
                steps.append(float(m2.group(2)))
    except Exception as e:
        print('  读取 %s 失败: %s' % (path, e))
    return vals, steps


A = os.path.join(ROOT, 'i5build', 'sgdrA.log')
B = os.path.join(ROOT, 'trainer_muon3.log')
va, ta = curve(A)
vb, tb = curve(B)

print('  A（AdamW）: %d 个验证点, %d 个步时' % (len(va), len(ta)))
print('  B（Muon+NS=3）: %d 个验证点, %d 个步时' % (len(vb), len(tb)))
if not va or not vb:
    print()
    print('  B 的验证点还没有 —— 1500 步仍在跑，稍后重跑本脚本即可。')
    sys.exit(0)

print()
print('  步数      AdamW      Muon+NS3     差')
for (sa, la), (sb, lb) in zip(va, vb):
    print('  %-8d %9.4f %11.4f  %+9.4f' % (sa, la, lb, lb - la))

if ta and tb:
    ma, mb = statistics.median(ta), statistics.median(tb)
    print()
    print('  每步耗时（中位）: AdamW %.3f s · Muon+NS3 %.3f s ⇒ 成本比 %.3f×'
          % (ma, mb, mb / ma))

# cumulative wall clock per validation checkpoint
def wall(times, upto):
    return sum(times[:upto]) if upto <= len(times) else None


print()
print('  到同一验证 loss 的步数与墙钟（平滑窗口 20，日志每 250 步一个点，故窗口取小）:')
def smooth(v, w=3):
    out = []
    for i in range(len(v)):
        lo = max(0, i - w + 1)
        out.append(sum(x[1] for x in v[lo:i + 1]) / (i - lo + 1))
    return out

sma, smb = smooth(va), smooth(vb)
print('    %-10s %10s %10s %12s %12s' % ('目标 val', 'A 步数', 'B 步数', 'A 墙钟 s', 'B 墙钟 s'))
for target in (6.4, 6.2, 6.0, 5.9, 5.85, 5.8):
    na = next((va[i][0] for i, x in enumerate(sma) if x <= target), None)
    nb = next((vb[i][0] for i, x in enumerate(smb) if x <= target), None)
    wa_ = None if na is None or not ta else sum(ta[:na])
    wb_ = None if nb is None or not tb else sum(tb[:nb])
    print('    %-10.2f %10s %10s %12s %12s'
          % (target, na or '未达到', nb or '未达到',
             '%.0f' % wa_ if wa_ else '—', '%.0f' % wb_ if wb_ else '—'))

print()
print('  两条保留:  1) 只有验证 loss，没有生成质量评估。')
print('             2) 单种子。计划要求 >=3，这是筛选。')
