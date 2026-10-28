# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import io, re
s = io.open(r'D:\work\bitsandbytes-CPU\train_ar_v2.py', encoding='utf-8').read()
i = s.find('def build_optimizers')
print(s[i:i + 2600])
