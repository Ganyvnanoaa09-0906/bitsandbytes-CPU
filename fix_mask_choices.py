# -*- coding: utf-8 -*-
"""fix_mask_choices.py — 给 --mask-mode 的 choices 加 'full'"""
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
