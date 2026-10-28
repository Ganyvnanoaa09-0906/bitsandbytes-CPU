# -*- coding: utf-8 -*-
"""reject_hist.py — 统计真实丢弃原因分布（决定 Seedream 式回收能拿回多少）"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import json, os, io, sys
from collections import Counter

ROOT = r'D:\work\bitsandbytes-CPU\.autopilot'
DIRS = ['clean_nf', 'clean_dl', 'clean_char2_1024']

print('=' * 88)
print('丢弃原因分布（真实数据）')
print('=' * 88)
grand = Counter()
tot = 0
for d in DIRS:
    p = os.path.join(ROOT, d, 'rejected.jsonl')
    if not os.path.exists(p):
        print('  %-22s 无 rejected.jsonl' % d); continue
    c = Counter()
    n = 0
    for line in io.open(p, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        n += 1
        for reason in (r.get('reasons') or ['?']):
            c[reason] += 1
    print()
    print('  --- %s : 丢弃 %d 张 ---' % (d, n))
    print('  %8s %7s  %s' % ('张数', '占比', '原因'))
    for k, v in c.most_common():
        print('  %8d %6.1f%%  %s' % (v, 100 * v / max(n, 1), k))
    grand += c
    tot += n

# 各 clean 目录保留了多少
print()
print('  === 保留量对比 ===')
for d in DIRS:
    rp = os.path.join(ROOT, d, 'clean_report.json')
    np_ = len(os.listdir(os.path.join(ROOT, d, 'images'))) if os.path.isdir(os.path.join(ROOT, d, 'images')) else 0
    info = json.load(io.open(rp, encoding='utf-8')) if os.path.exists(rp) else {}
    print('    %-22s 保留 %6d   报告 %s' % (d, np_, json.dumps(info, ensure_ascii=False)))

print()
print('=' * 88)
print('合计丢弃 %d 张，原因分布：' % tot)
for k, v in grand.most_common():
    print('  %8d %6.1f%%  %s' % (v, 100 * v / max(tot, 1), k))

print()
print('可回收性判读（对照 Seedream 3.0 的"缺陷面积<20% 就保留 + latent 屏蔽"）:')
RECOVER = {
    'no_person': '内容过滤 —— 只有当目标分布确实要人像时才该丢；若要通用图则可全回收',
    'too_many_person': '同上，人数上限是人为设定',
    'scenery': '内容过滤 —— 可回收（只是分布偏移）',
    'food': '内容过滤 —— 可回收',
    'screenshot': '内容过滤 —— 可回收（但会引入 UI 文字分布）',
    'blur': '质量硬伤 —— Seedream 的 latent 屏蔽【不适用】（整图都糊）',
    'too_small': '分辨率硬伤 —— 无法回收',
    'plain': '低方差（纯色）—— 无法回收',
    'decode_error': '损坏 —— 无法回收',
}
for k, v in grand.most_common():
    print('  %-18s %s' % (k, RECOVER.get(k, '未分类')))
print('DONE')
