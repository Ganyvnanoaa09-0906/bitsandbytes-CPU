# -*- coding: utf-8 -*-
"""v3_clean_speed.py — 从 v3 日志里挖出【未被其他测试干扰】的单步耗时。

v3 的 T=256 阶段单步耗时在不同时间段差别很大：
  阶段刚开始时机器空闲 → 干净
  我做核显/并发测试时 → 被抢核，失真

按时间区间分段统计，找出干净区间。
"""
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
import json, os
from collections import defaultdict

paths = [
    (r"D:\work\bitsandbytes-CPU\.autopilot\v3_noloop\train.jsonl", "原始运行"),
    (r"D:\work\bitsandbytes-CPU\.autopilot\v3_noloop\train.jsonl", None),
]
p = paths[0][0]
rows = []
for line in open(p, encoding='utf-8', errors='ignore'):
    line = line.strip()
    if line.startswith('{'):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

print('日志 %d 行' % len(rows))

T256 = [r for r in rows if r.get('T') == 256]
print('T=256 阶段 %d 条' % len(T256))
if not T256:
    raise SystemExit('没有 T=256 数据')

# 按 elapsed_s 分桶（elapsed 是本次进程内的秒数，重启会归零）
print()
print('--- T=256 单步耗时随时间 ---')
print('  %-10s %-10s %10s %10s' % ('step起', 'step止', 'step_s均值', 'tok/s'))
ts = [r.get('step_s') for r in T256 if r.get('step_s')]
steps = [r.get('step') for r in T256]
k = max(1, len(ts) // 10)
for i in range(0, len(ts), k):
    seg = ts[i:i + k]
    if not seg:
        continue
    avg = sum(seg) / len(seg)
    print('  %-10s %-10s %10.3f %10.1f'
          % (steps[i], steps[min(i + k - 1, len(steps) - 1)], avg, 8 * 256 / avg))

best = min(ts)
worst = max(ts)
print()
print('  最快 %.3f s/步   最慢 %.3f s/步   相差 %.1f×' % (best, worst, worst / best))

# 建模：干净单步 FLOPs
B, T, NL, D, FFN, NH, NKV, HD = 8, 256, 10, 512, 2048, 8, 4, 64
per_layer = (2 * B * T * D * (NH * HD + 2 * NKV * HD)      # QKV
             + 2 * B * T * D * D                            # o
             + 3 * 2 * B * T * D * FFN                      # SwiGLU (w1,w3,w2)
             + 2 * 2 * B * NH * T * T * HD)                 # attention QK^T + AV
fwd = per_layer * NL
tot = fwd * 3

print()
print('--- 折算成 GFLOPS ---')
print('  每层 %.2f GFLOP   前向 %.1f GFLOP   训练(x3) %.1f GFLOP' % (per_layer / 1e9, fwd / 1e9, tot / 1e9))
for tag, s in (('最快(最干净)', best), ('最慢(被抢核)', worst)):
    print('  %-14s %.3f s/步  ->  %.1f GFLOPS' % (tag, s, tot / s / 1e9))
print()
print('  对照: 持续 sgemm 稳态 94.87 GFLOPS (n=1024) / 118~120 (n=2048)')
