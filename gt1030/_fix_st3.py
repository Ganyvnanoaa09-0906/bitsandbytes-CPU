# -*- coding: utf-8 -*-
"""selftest 里对拍时把 device 梯度除回 CE_SCALE（梯度是放大过的）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi_selftest.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""    def cmp(name, a, b):
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)""",
"""    SC = MM.CE_SCALE

    def cmp(name, a, b):
        a = np.asarray(a, np.float32) / SC; b = np.asarray(b, np.float32)""")

# 所有 device 侧实参（第一个）保持原样 —— 在 cmp 里统一除，足够
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched selftest compare')
