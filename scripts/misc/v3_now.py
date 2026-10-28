# -*- coding: utf-8 -*-
"""v3_now.py — v3 当前实时速度与进度。"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import json, time, os
from datetime import datetime

p = r"D:\work\bitsandbytes-CPU\.autopilot\v3_noloop\train.jsonl"
rows = []
for line in open(p, encoding='utf-8', errors='ignore'):
    line = line.strip()
    if line.startswith('{'):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

print('日志 %d 行   文件更新时间 %s'
      % (len(rows), datetime.fromtimestamp(os.path.getmtime(p)).strftime('%H:%M:%S')))

T256 = [r for r in rows if r.get('T') == 256]
print('T=256 阶段 %d 条' % len(T256))
if T256:
    print()
    print('--- 最近 12 条 ---')
    print('  %-8s %-10s %-10s %-10s' % ('全局step', 'loss', 's/步', 'tok/s'))
    for r in T256[-12:]:
        ss = r.get('step_s', 0)
        print('  %-8s %-10.4f %-10.3f %-10.1f'
              % (r.get('step'), r.get('loss', 0), ss, 8 * 256 / ss if ss else 0))

    recent = [r['step_s'] for r in T256[-20:] if r.get('step_s')]
    if recent:
        avg = sum(recent) / len(recent)
        print()
        print('  最近 20 条均值: %.3f s/步   %.1f tok/s' % (avg, 8 * 256 / avg))
        print('  (干净基线是 2.31~2.37 s / 886 tok/s)')
    last = T256[-1].get('step')
    total = 30000
    remain = (total - last) * (sum(recent) / len(recent) if recent else 2.4) / 3600
    print()
    print('  进度: 全局 step %s / %d   剩余约 %.1f 小时' % (last, total, remain))
