# -*- coding: utf-8 -*-
"""fix_help_percent.py — 修 train_ar_v2.py 里 help 字符串中未转义的 %

预先存在的 bug：--head-mode 的 help 里有 "1.7%，" 和 "1.6% 却"，
argparse 会做 help % params ⇒ ValueError: unsupported format character
⇒ --help 从来没能跑通（训练不调用 help formatter，所以一直没暴露）

修法：help 字符串里的字面 % 必须写成 %%
"""
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

fixes = [
    ("    help='flat 是本规模下的正确选择：输出层只占总算力 1.7%，'",
     "    help='flat 是本规模下的正确选择：输出层只占总算力 1.7%%，'"),
    ("                         '因子化只省 1.6% 却把 1024 类压成 32x32 可分解近似（实测劣于 flat）')",
     "                         '因子化只省 1.6%% 却把 1024 类压成 32x32 可分解近似（实测劣于 flat）')"),
]
n = 0
for a, b in fixes:
    if a in s:
        s = s.replace(a, b, 1)
        n += 1
    else:
        print('未找到（可能已修）:', a[:50])

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('修了 %d 处' % n)

# 全量扫描：找 help= 里未转义的 %
import re
bad = []
for i, line in enumerate(s.split('\n'), 1):
    if 'help=' in line or (line.strip().startswith("'") and i > 1):
        # 检查是否有单个 % 后跟非格式字符
        for m in re.finditer(r'%(?![%sdfegrx\(\-\.0-9])', line):
            bad.append((i, line.strip()[:80]))
print()
print('剩余可疑的行:')
for i, t in bad:
    print('  L%d: %s' % (i, t))
if not bad:
    print('  （无）')
