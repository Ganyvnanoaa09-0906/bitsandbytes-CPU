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
import io, re
s = io.open(r'_os.path.join(_root, 'scripts\\lib\\train_ar_v2.py')', encoding='utf-8').read()
# find the actual optimizer object creation
for m in re.finditer(r'\n\s*opt\w*\s*=\s*[^\n]+', s):
    print(m.group(0).strip())
print('--- betas ---')
for m in re.finditer(r'betas\s*=\s*\([^)]*\)', s):
    print(m.group(0))
print('--- AdamW( ---')
for m in re.finditer(r'AdamW\(', s):
    a = max(0, m.start() - 300)
    print('...', s[a:m.end() + 260].replace('\n', ' | ')[-460:])
    print()
