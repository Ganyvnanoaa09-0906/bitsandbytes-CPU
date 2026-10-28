# -*- coding: utf-8 -*-
"""analyze_dups.py - 把 final_dups.jsonl 按「池 -> 池」归类，判断重复来自哪个源。"""
from __future__ import annotations
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
import collections, json, os

BASE = r'D:\work\bitsandbytes-CPU\.autopilot'
POOLS = [
    ('char2',     r'clean_char2_1024\images'),
    ('dl',        r'clean_dl\images'),
    ('webp',      r'clean_webp\images'),
    ('photobook', r'sujvji\photobook'),
    ('tu',        r'clean_tu\images'),
    ('za',        r'clean_za\images'),
]


def pool_of(p):
    for name, frag in POOLS:
        if frag in p:
            return name
    return '?'


def setname(p):
    """取所属集目录名（char2 这类扁平池返回其父目录名）。"""
    d = os.path.dirname(p)
    return os.path.basename(d)


def main():
    rows = [json.loads(l) for l in open(os.path.join(BASE, 'final_dups.jsonl'), encoding='utf-8')
            if l.strip()]
    print('重复组总数: %d' % len(rows))

    pair2 = collections.Counter()      # n==2 的定向对
    poolsets = collections.Counter()   # 多图组的池集合
    dup_as = collections.Counter()     # 每个池被当成 dup 移除的张数
    keep_as = collections.Counter()    # 每个池被保留的张数
    moved_pairs = collections.Counter()  # (keep池, dup池) 不计方向

    for r in rows:
        kp = pool_of(r['keep'])
        keep_as[kp] += 1
        dps = [pool_of(d) for d in r['dup']]
        for d in dps:
            dup_as[d] += 1
        if len(r['dup']) == 1:
            pair2[(kp, dps[0])] += 1
            moved_pairs[tuple(sorted((kp, dps[0])))] += 1
        else:
            poolsets[tuple(sorted(set([kp] + dps)))] += 1

    print()
    print('=== 重复的来源（keep 池 -> 被移除 池），仅 n=2 的组 ===')
    print('%-12s %-12s %8s %10s' % ('keep 池', 'dup 池', '组数', '移除张数'))
    for (a, b), c in pair2.most_common(20):
        print('%-12s %-12s %8d %10d' % (a, b, c, c))

    print()
    print('=== 每个池被移除 / 被保留 的张数 ===')
    allp = sorted(set(list(dup_as) + list(keep_as)))
    print('%-12s %10s %10s' % ('池', '被移除', '被保留'))
    for p in allp:
        print('%-12s %10d %10d' % (p, dup_as[p], keep_as[p]))

    print()
    print('=== 多图重复组（n>=3）涉及的池组合 ===')
    for k, c in poolsets.most_common(10):
        print('  %6d 组   %s' % (c, ' + '.join(k)))

    print()
    print('=== clean_webp 被移除的图来自哪些集（top 15）===')
    sets = collections.Counter()
    for r in rows:
        for d in r['dup']:
            if r'clean_webp' in d:
                sets[setname(d)] += 1
    for k, c in sets.most_common(15):
        print('  %6d  %s' % (c, k))
    print('  ... 共涉及 %d 个集' % len(sets))

    print()
    print('=== 抽样 8 组跨池重复（路径对照）===')
    shown = 0
    for r in rows:
        kp = pool_of(r['keep'])
        dp = pool_of(r['dup'][0])
        if kp != dp and kp in ('webp', 'char2') and shown < 8:
            print('  [%s] %s' % (kp, r['keep']))
            print('  [%s] %s' % (dp, r['dup'][0]))
            print('  ----')
            shown += 1

    print()
    print('=== 最大重复组 ===')
    big = max(rows, key=lambda r: r['n'])
    print('  n = %d' % big['n'])
    print('  keep: %s' % big['keep'])
    for d in big['dup'][:12]:
        print('    dup: %s' % d)
    print('    ...')


if __name__ == '__main__':
    main()
