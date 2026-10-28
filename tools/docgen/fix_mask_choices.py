# -*- coding: utf-8 -*-
"""fix_mask_choices.py — 给 --mask-mode 的 choices 加 'full'"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import io

P = r'D:\work\bitsandbytes-CPU\train_ar_v2.py'
s = io.open(P, encoding='utf-8-sig').read()

OLD = "                    choices=['causal', 'block'],"
NEW = "                    choices=['causal', 'block', 'full'],"
if "'full']" in s:
    print('已包含 full')
else:
    assert OLD in s, '找不到 choices'
    s = s.replace(OLD, NEW, 1)
    print("已加 'full' choice")

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('行数', s.count('\n') + 1)
