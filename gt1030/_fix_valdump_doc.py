# -*- coding: utf-8 -*-
# 修 ar_v2_valdump.py 的 docstring：里面有 F:\\weibo_cos\\valdump\\xxx.npz，\v \x 会被当转义。
import io
import py_compile

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_valdump.py'
s = io.open(p, encoding='utf-8').read()
if s.startswith('"""'):
    s = 'r' + s
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
    print('docstring -> raw')
py_compile.compile(p, doraise=True)
print('compile ok')
