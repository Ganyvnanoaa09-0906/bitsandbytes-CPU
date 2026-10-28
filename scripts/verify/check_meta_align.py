# -*- coding: utf-8 -*-
"""check_meta_align.py — 确认 metadata.jsonl 与 token 清单能否对齐

条件注入的前提是每张图的元数据能按路径查到。这里只比对【路径格式与命中率】，
不打印任何图片内容或统计值细节。
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

meta = r'.autopilot\clean_nf\metadata.jsonl'
paths_json = r'.autopilot\tokens_final2.pt.paths.json'

print('=== metadata.jsonl 的 image 字段格式 ===')
n = 0
samples = []
fields_seen = set()
for line in open(meta, encoding='utf-8', errors='ignore'):
    line = line.strip()
    if not line.startswith('{'):
        continue
    r = json.loads(line)
    n += 1
    fields_seen.update(r.keys())
    if len(samples) < 3:
        samples.append((r.get('image', ''), r.get('src', '')))
print('  总条数:', n)
print('  字段:', sorted(fields_seen))
for a, b in samples:
    print('  image 字段样式: %r' % (a[:110],))
    print('  src   字段样式: %r' % (str(b)[:90],))

print()
print('=== token 清单格式 ===')
try:
    pj = json.load(open(paths_json, encoding='utf-8'))
    print('  条目数:', len(pj))
    for x in pj[:3]:
        print('   %r' % (str(x)[:110],))
except FileNotFoundError:
    print('  不存在:', paths_json)

print()
print('=== 命中率测试（用 abspath 归一化）===')
try:
    pj = json.load(open(paths_json, encoding='utf-8'))
    by = {}
    for line in open(meta, encoding='utf-8', errors='ignore'):
        line = line.strip()
        if not line.startswith('{'):
            continue
        r = json.loads(line)
        im = r.get('image', '')
        if im:
            by[os.path.abspath(im)] = r
            by[os.path.basename(im)] = r
    hit_abs = sum(1 for p in pj if os.path.abspath(p) in by)
    hit_base = sum(1 for p in pj if os.path.basename(p) in by)
    print('  用 abspath 命中 %d/%d (%.1f%%)' % (hit_abs, len(pj), 100 * hit_abs / len(pj)))
    print('  用 basename 命中 %d/%d (%.1f%%)' % (hit_base, len(pj), 100 * hit_base / len(pj)))
    print('  ⇒ 若两者都低，说明 metadata 覆盖的不是这批图')
except Exception as e:
    print('  失败:', type(e).__name__, str(e)[:120])
