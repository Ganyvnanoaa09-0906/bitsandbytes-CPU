# -*- coding: utf-8 -*-
"""v3_speed_trend.py — 从 v3 的训练日志里挖出「速度随时间的变化」。

这是最真实的降频证据：真实训练负载、真实散热条件、连续几小时。
如果 tokens_per_s 随时间下滑，说明撞了功耗墙。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import json, os

p = r"D:\work\bitsandbytes-CPU\.autopilot\v3_noloop\train.jsonl"
rows = []
for line in open(p, encoding='utf-8', errors='ignore'):
    line = line.strip()
    if line.startswith('{'):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

print('v3 日志 %d 行' % len(rows))
if not rows:
    raise SystemExit

# 按 T 阶段分组（不同 T 的 tok/s 不可比）
from collections import OrderedDict
stages = OrderedDict()
for r in rows:
    stages.setdefault(r.get('T'), []).append(r)

for T, rs in stages.items():
    print()
    print('===== T=%d (%d 条记录) =====' % (T, len(rs)))
    ts = [r.get('tokens_per_s', 0) for r in rs if r.get('tokens_per_s')]
    if not ts:
        print('  (无 tok/s)')
        continue
    # 分 6 段看趋势
    k = max(1, len(ts) // 6)
    print('  %-16s %10s %10s' % ('区间', '均值 tok/s', '相对首段'))
    first = None
    for i in range(0, len(ts), k):
        seg = ts[i:i + k]
        if not seg:
            continue
        avg = sum(seg) / len(seg)
        if first is None:
            first = avg
        step_lo = rs[i].get('step')
        step_hi = rs[min(i + k - 1, len(rs) - 1)].get('step')
        print('  step %5s-%-5s %10.1f %9.1f%%' % (step_lo, step_hi, avg, 100 * avg / first))
    print('  全局: 最快 %.1f  最慢 %.1f  衰减 %.1f%%'
          % (max(ts), min(ts), 100 * (max(ts) - min(ts)) / max(ts)))
