# -*- coding: utf-8 -*-
"""精确补 --warmup/--lr-min（上次锚点缩进对不上）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_train.py'
t = io.open(p, encoding='utf-8').read()
assert '--warmup' not in t, '已经有了'
old = "    ap.add_argument('--val-frac', type=float, default=0.02,"
new = ("    ap.add_argument('--warmup', type=int, default=0, help='前多少步线性升温到 --lr')\n"
       "    ap.add_argument('--lr-min', type=float, default=0.0, help='余弦衰减终点')\n"
       + old)
assert old in t
t = t.replace(old, new, 1)
io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
print('args added')
