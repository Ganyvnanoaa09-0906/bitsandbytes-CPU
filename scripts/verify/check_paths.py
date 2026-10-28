# -*- coding: utf-8 -*-
"""check_paths.py — 检查 token 文件的路径清单能不能在本地用"""
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
import json, io, os, sys

p = sys.argv[1] if len(sys.argv) > 1 else \
    r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt.paths.json'
d = json.load(io.open(p, encoding='utf-8'))
print('条目数 %d' % len(d))
print('前 3 条:')
for x in d[:3]:
    print('  ' + x)
print('末 2 条:')
for x in d[-2:]:
    print('  ' + x)
ex = sum(1 for x in d[:300] if os.path.exists(x))
print('前 300 条本地存在: %d' % ex)
# 尝试把 /workspace/bundle/data/ 映射到本地
alt_root = r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf'
def remap(x):
    if x.startswith('/workspace/bundle/data/'):
        return os.path.join(alt_root, x[len('/workspace/bundle/data/'):].replace('/', os.sep))
    return x
ex2 = sum(1 for x in d[:300] if os.path.exists(remap(x)))
print('前 300 条经 /workspace/bundle/data -> clean_nf 映射后存在: %d' % ex2)
if ex2:
    print('  映射示例:'); print('    ' + d[0]); print('    -> ' + remap(d[0]))
