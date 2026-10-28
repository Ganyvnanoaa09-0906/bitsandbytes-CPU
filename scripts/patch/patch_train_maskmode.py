# -*- coding: utf-8 -*-
"""patch_train_maskmode2.py — 给 train_ar_v2.py 加 --mask-mode flag（修正版）

上一版找不到锚点，因为 flag 用的是横线（--pos-mode）不是下划线。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import io

P = r'D:\work\bitsandbytes-CPU\train_ar_v2.py'
s = io.open(P, encoding='utf-8-sig').read()
changed = []

# ① argparse：插在 --token-order 之后
if "'--mask-mode'" not in s:
    anchor = "    ap.add_argument('--token-order', default='raster', choices=['raster', 'zorder'])"
    assert anchor in s, '找不到 --token-order'
    j = s.find(anchor) + len(anchor)
    ins = ("\n    ap.add_argument('--mask-mode', default='causal',\n"
           "                    choices=['causal', 'block'],\n"
           "                    help='causal = 光栅序逐 token 因果（默认，向后兼容）；'\n"
           "                         'block = 借 VAR(arXiv 2404.02905, NeurIPS 2024 Best Paper) '\n"
           "                         '的掩码结构：Z-order 块内双向、块间因果。'\n"
           "                         '动机：实测 T=256 光栅序 causal 下生成 token 相邻同率 '\n"
           "                         '42~54%%（真实 4.47%%）且逐位置 top-1 越后越差 '\n"
           "                         '(11.5%%→5.7%%)，即 token 坍塌 + 长程依赖失效。'\n"
           "                         '⚠️ block 必须配合 --token-order zorder '\n"
           "                         '（4^k 前缀 = 2^k x 2^k 方形子块）。')")
    s = s[:j] + ins + s[j:]
    changed.append('argparse')

# ② 传给 cfg
if 'mask_mode=a.mask_mode' not in s:
    anchor = "pos_mode=a.pos_mode, token_order=a.token_order,"
    assert anchor in s, '找不到 cfg 构造点'
    s = s.replace(anchor, anchor + "\n                             mask_mode=a.mask_mode,", 1)
    changed.append('cfg')

# ③ 启动时打印
if '[mask]' not in s:
    anchor = "    print('[v2] params="
    i = s.find(anchor)
    assert i >= 0, '找不到打印点'
    ins = ("    if getattr(a, 'mask_mode', 'causal') == 'block':\n"
           "        print('[mask] VAR 风格 block-causal：Z-order 块内双向/块间因果  '\n"
           "              'token_order=%s  自回归步数 = log4(T)+1' % a.token_order, flush=True)\n"
           "        if a.token_order != 'zorder':\n"
           "            print('[mask] ⚠️ block mask 要求 token_order=zorder，当前是 %s'\n"
           "                  % a.token_order, flush=True)\n")
    s = s[:i] + ins + s[i:]
    changed.append('print')

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('已修改:', changed if changed else '（无需修改）')
print('行数:', s.count('\n') + 1)
