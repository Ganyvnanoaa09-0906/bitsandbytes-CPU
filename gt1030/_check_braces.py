# -*- coding: utf-8 -*-
"""定位 sector_mirror.c 的括号失衡并报告（只读）。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_mirror.c'
lines = io.open(p, encoding='utf-8').read().splitlines()
d = 0
for i, ln in enumerate(lines, 1):
    t = ln
    # 去掉 C 字符串字面量（成对引号间的部分）
    while True:
        a = t.find('"')
        if a < 0:
            break
        b = t.find('"', a + 1)
        if b < 0:
            t = t[:a]
            break
        t = t[:a] + t[b + 1:]
    # 去掉行注释
    c = t.find('//')
    if c >= 0:
        t = t[:c]
    d += t.count('{') - t.count('}')
    if d < 0:
        print('depth<0 at line %d: %s' % (i, ln.strip()[:90]))
        break
print('final depth =', d)
if d != 0:
    for i, ln in enumerate(lines, 1):
        if 255 <= i <= 300 or 330 <= i <= 389:
            print('%3d| %s' % (i, ln[:100]))
