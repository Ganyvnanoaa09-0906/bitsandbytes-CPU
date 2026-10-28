# -*- coding: utf-8 -*-
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
import io
s = io.open(r'_os.path.join(_root, 'scripts\\lib\\train_ar_v2.py')', encoding='utf-8').read()
i = s.find('torch.optim.AdamW')
print(s[max(0, i - 900):i + 800])
print('=== --opt flag ===')
j = s.find("'--opt'")
print(s[j - 100:j + 400] if j > 0 else 'no --opt')
