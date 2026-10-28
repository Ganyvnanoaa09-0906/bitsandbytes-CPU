# -*- coding: utf-8 -*-
"""upload_b64.py — 把文件转成 base64 文本文件（供 scp 传输，ASCII 安全）"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import base64
import os

SRC = r'D:\work\bitsandbytes-CPU'
OUT = r'D:\work\b64up'
os.makedirs(OUT, exist_ok=True)

FILES = ['small_image_model_v2.py', 'train_ar_v2.py', 'grn_refine.py', 'bitwise_head.py']

for f in FILES:
    p = os.path.join(SRC, f)
    if not os.path.exists(p):
        print('%-28s 缺失' % f)
        continue
    data = open(p, 'rb').read()
    b = base64.b64encode(data).decode('ascii')
    txt = os.path.join(OUT, f + '.b64')
    with open(txt, 'w', encoding='ascii', newline='\n') as fh:
        for i in range(0, len(b), 76):
            fh.write(b[i:i + 76] + '\n')
    print('%-28s %6.1f KB -> %s' % (f, len(data) / 1024, os.path.basename(txt)))
