"""Compare the LR-fixed Muon run against the AdamW baseline and against the previous Muon run.

Three curves now, all through train_ar_v2.py under identical settings (1500 steps, batch 2,
T=1024, validation every 250 on the same held-out set):

    AdamW                    i5build/sgdrA.log
    Muon+NS=3, cosine LR     trainer_muon3.log      (lost by ~0.5 at every point)
    Muon+NS=3, pinned LR     trainer_muon3_lrfix.log  (this run)

The middle one is kept deliberately. If the LR fix is the explanation, the difference between
the second and third rows is the evidence, and without the second row there would be nothing to
attribute the change to.

Reported: validation at each checkpoint for all three, median step cost, and steps and wall
clock to reach a given loss. Both stated limitations carry over: validation loss only, single
seed.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
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
    vals, steps = [], []
    try:
        for ln in io.open(path, encoding='utf-8', errors='replace'):
            m = re.search(r'step (\d+)\s+train=[\d.]+\s+val=([\d.]+)', ln)
            if m:
                vals.append((int(m.group(1)), float(m.group(2))))
            m2 = re.search(r'(\d+)/1500 loss=[\d.]+ ([\d.]+)s', ln)
            if m2:
                steps.append(float(m2.group(1 + 1)))
    except Exception:
        pass
    return vals, steps


A = os.path.join(ROOT, 'i5build', 'sgdrA.log')
B = os.path.join(ROOT, 'trainer_muon3.log')
C = os.path.join(ROOT, 'trainer_muon3_lrfix.log')

va, ta = curve(A)
vb, tb = curve(B)
vc, tc = curve(C)

print('  AdamW 臂:        %d 个验证点' % len(va))
print('  Muon 退火 lr:    %d 个验证点' % len(vb))
print('  Muon 固定 lr:    %d 个验证点' % len(vc))
if not vc:
    print()
    print('  固定 lr 的那次还没有验证点 —— 1500 步仍在跑。')
    sys.exit(0)

print()
print('  步数      AdamW       Muon退火    Muon固定    固定-退火')
for i in range(max(len(va), len(vb), len(vc))):
    sa = va[i][0] if i < len(va) else None
    la = va[i][1] if i < len(va) else None
    lb = vb[i][1] if i < len(vb) else None
    lc = vc[i][1] if i < len(vc) else None
    print('  %-9s %10s %11s %11s %11s'
          % (sa or '—',
             '%.4f' % la if la else '—',
             '%.4f' % lb if lb else '—',
             '%.4f' % lc if lc else '—',
             '%+.4f' % (lc - lb) if (lc and lb) else '—'))

for name, ts in (('AdamW', ta), ('Muon 退火', tb), ('Muon 固定', tc)):
    if ts:
        print('  %s 每步中位: %.3f s' % (name, statistics.median(ts)))

print()
print('  到同一 val 的步数:')
print('    %-10s %10s %12s %12s' % ('目标', 'AdamW', 'Muon退火', 'Muon固定'))
for target in (6.6, 6.4, 6.2, 6.0, 5.9, 5.85, 5.8):
    def first(v):
        for s, x in v:
            if x <= target:
                return s
        return None
    print('    %-10.2f %10s %12s %12s'
          % (target, first(va) or '未达到', first(vb) or '未达到', first(vc) or '未达到'))

print()
print('  读法: 若固定 lr 后的曲线明显贴近 AdamW（或更好）⇒ 退火是之前落后的全部原因；')
print('        若仍然落后 ⇒ 训练器与 harness 的差异另有所在，需要继续找。')
print('  保留: 1) 只有验证 loss 2) 单种子')
