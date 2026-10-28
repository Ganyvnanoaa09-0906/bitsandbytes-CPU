# -*- coding: utf-8 -*-
"""patch_mask_mode.py — 在 SelfAttention.__init__ 里保存 _mask_mode，并去掉 BOM"""
import os, sys
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
import io

P = r'D:\work\bitsandbytes-CPU\small_image_model_v2.py'
s = io.open(P, encoding='utf-8-sig').read()      # utf-8-sig 自动吃掉 BOM

OLD = '        self.nh, self.n_kv = nh, cfg.n_kv_head'
NEW = ('        self.nh, self.n_kv = nh, cfg.n_kv_head\n'
       "        self._mask_mode = getattr(cfg, 'mask_mode', 'causal')")

if '_mask_mode = getattr' in s:
    print('_mask_mode 已存在，跳过')
else:
    assert OLD in s, '找不到锚点:\n' + OLD
    s = s.replace(OLD, NEW, 1)
    print('已插入 _mask_mode')

io.open(P, 'w', encoding='utf-8', newline='').write(s)   # 无 BOM，LF
print('已写回（无 BOM）')
print('行数:', s.count('\n') + 1)
