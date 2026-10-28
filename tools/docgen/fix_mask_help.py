# -*- coding: utf-8 -*-
"""fix_mask_help.py — 去掉 --mask-mode help 里所有 %（argparse 会做 help % params）"""
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

P = r'_os.path.join(_root, 'scripts\\lib\\train_ar_v2.py')'
s = io.open(P, encoding='utf-8-sig').read()

start = s.find("    ap.add_argument('--mask-mode'")
assert start >= 0, '找不到 --mask-mode'
# 找到这个 add_argument 调用的结尾（下一个顶层 "    ap.add_argument" 或空行 + 非续行）
end = s.find("\n    ap.add_argument(", start + 10)
assert end > start, '找不到结尾'
new = ("    ap.add_argument('--mask-mode', default='causal',\n"
       "                    choices=['causal', 'block'],\n"
       "                    help='causal = 光栅序逐 token 因果（默认，向后兼容）；'\n"
       "                         'block = 借 VAR(arXiv 2404.02905, NeurIPS 2024 Best Paper) '\n"
       "                         '的掩码结构：Z-order 块内双向、块间因果。'\n"
       "                         '动机：实测 T=256 光栅序 causal 下生成 token 相邻同率高'\n"
       "                         '（真实值 4.47 的 9 到 12 倍）且逐位置 top-1 越后越差，'\n"
       "                         '即 token 坍塌加长程依赖失效。'\n"
       "                         '注意：block 必须配合 --token-order zorder。')")
s = s[:start] + new + s[end:]

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('已替换 --mask-mode 的 help（去掉所有 %）')
# 打印确认
i = s.find("'--mask-mode'")
print(s[i - 30:i + 700])
