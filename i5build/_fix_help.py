"""修 argparse help 里的 % 转义（旧版 Python 不校验，新版校验 ⇒ 只有 i5 崩）。

只动两处未转义的：L289 的 '5%'、L296 的 '79.6%'。
其余 9 处已经是 %% 了，别碰（碰了会显示成 %%%）。
"""
import ast
import sys

sys.stdout.reconfigure(encoding='utf-8')
p = r'D:\work\bitsandbytes-CPU\train_ar_v2.py'
s = open(p, encoding='utf-8').read()

FIX = [('【新文件尾部 5%】', '【新文件尾部 5%%】'),
       ('top-1 79.6% 全是假数字', 'top-1 79.6%% 全是假数字')]
n = 0
for old, new in FIX:
    if new in s:
        print('  已经是转义态，跳过:', old)
        continue
    if old in s:
        s = s.replace(old, new, 1)
        n += 1
        print('  ✓', old, '→', new)
    else:
        print('  ✗ 没找到:', old)

if n:
    ast.parse(s)
    open(p, 'w', encoding='utf-8').write(s)
    print('已改 %d 处，语法 OK' % n)
else:
    print('没有需要改的')
