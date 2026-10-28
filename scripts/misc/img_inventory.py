# -*- coding: utf-8 -*-
"""img_inventory.py — 准确统计可用的图片数据（按目录分组）"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys
from collections import Counter

ROOT = r'D:\work'
EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
SKIP = ('node_modules', 'site-packages', '.git', 'docs_cpu')

cnt = Counter()
total = 0
for dp, dns, fns in os.walk(ROOT):
    if any(s in dp for s in SKIP):
        dns[:] = []
        continue
    n = sum(1 for f in fns if f.lower().endswith(EXTS))
    if n:
        rel = os.path.relpath(dp, ROOT)
        # 取前两级作为分组键
        parts = rel.split(os.sep)
        key = os.sep.join(parts[:2]) if len(parts) >= 2 else rel
        cnt[key] += n
        total += n

print('总计 %d 张图（排除 node_modules/site-packages/.git/docs_cpu）' % total)
print()
print('  %8s  %s' % ('数量', '目录（前两级）'))
print('  ' + '-' * 72)
for k, v in cnt.most_common(25):
    print('  %8d  %s' % (v, k))

# 去掉已经用上的那 29113 张之后，还剩什么
USED = ('clean_nf', 'clean_dl', 'clean_char2_1024')
print()
print('=== 已经在用（29,113 张）之外的目录 ===')
extra = 0
for k, v in cnt.most_common():
    if any(u in k for u in USED):
        continue
    print('  %8d  %s' % (v, k))
    extra += v
print('  ' + '-' * 72)
print('  未使用的图片总数: %d' % extra)
