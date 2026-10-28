# -*- coding: utf-8 -*-
"""cmp_ab.py — 对比 v2_run2(loop_times=2) 与 v3_noloop(loop_times=1) 的 loss 曲线。

两者同 seed(2024)、同数据、同超参，唯一差别是循环块。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import json, os

BASE = r"D:\work\bitsandbytes-CPU\.autopilot"


def load(p):
    rows = []
    if not os.path.exists(p):
        print('  (缺 %s)' % p)
        return rows
    for line in open(p, encoding='utf-8', errors='ignore'):
        line = line.strip()
        if line.startswith('{'):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


a = load(os.path.join(BASE, 'v2_run2', 'train.jsonl'))
b = load(os.path.join(BASE, 'v3_noloop', 'train.jsonl'))
print('v2_run2(loop=2): %d 行   v3_noloop(loop=1): %d 行' % (len(a), len(b)))
if b:
    print('v3 最后: step=%s T=%s loss=%.4f' % (b[-1].get('step'), b[-1].get('T'), b[-1].get('loss')))
if a:
    print('v2 最后: step=%s T=%s loss=%.4f' % (a[-1].get('step'), a[-1].get('T'), a[-1].get('loss')))


def near(rows, step, T, tol=40):
    best, bd = None, 10 ** 9
    for r in rows:
        if r.get('T') != T:
            continue
        d = abs(r.get('step', -999) - step)
        if d < bd:
            bd, best = d, r
    return best if bd <= tol else None


print()
print('%8s | %14s %14s | %10s' % ('step', 'v2 loop=2', 'v3 loop=1', 'v3 - v2'))
print('-' * 56)
for T in (16, 64, 256):
    print('--- T=%d ---' % T)
    for target in (200, 500, 1000, 1500, 2000, 2500, 3000):
        ra, rb = near(a, target, T), near(b, target, T)
        sa = '%14.4f' % ra['loss'] if ra else '%14s' % '-'
        sb = '%14.4f' % rb['loss'] if rb else '%14s' % '-'
        sd = '%+10.4f' % (rb['loss'] - ra['loss']) if (ra and rb) else ''
        print('%8d | %s %s | %s' % (target, sa, sb, sd))
