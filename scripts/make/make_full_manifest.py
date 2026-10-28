# -*- coding: utf-8 -*-
"""make_full_manifest.py — 从 tokens_final2 的路径清单构建全量图片清单

背景：本地还有 29,113 张图（tokens_final2 的 31,017 条里 93.9% 仍存在），
但 AR 只用的是 clean_nf 的 14,433 张。而验证集 gap 已从 +0.04 涨到 +0.51
（开始过拟合）⇒ 把数据翻倍是最直接的解法。

输出 jsonl，每行 {"path": "..."}，只保留文件确实存在的。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import json, os

src = r'.autopilot\tokens_final2.pt.paths.json'
dst = r'.autopilot\full_manifest.jsonl'

paths = json.load(open(src, encoding='utf-8'))
seen = set()
ok = 0
with open(dst, 'w', encoding='utf-8') as f:
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.exists(p):
            f.write(json.dumps({'path': p}, ensure_ascii=False) + '\n')
            ok += 1

print('  源清单 %d 条' % len(paths))
print('  去重后存在 %d 条' % ok)
print('  -> %s  (%.1f MB)' % (dst, os.path.getsize(dst) / 1e6))

# 顺便统计按来源目录的分布
import collections
c = collections.Counter(os.path.dirname(p) for p in paths if os.path.exists(p))
print()
print('  按来源目录（前 6）:')
for d, n in c.most_common(6):
    print('    %-64s %5d' % (d.replace('D:\\work\\bitsandbytes-CPU\\.autopilot\\', '')[:64], n))
