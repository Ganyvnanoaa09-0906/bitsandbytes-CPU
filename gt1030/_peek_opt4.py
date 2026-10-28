# -*- coding: utf-8 -*-
import io, re
s = io.open(r'D:\work\bitsandbytes-CPU\train_ar_v2.py', encoding='utf-8').read()
i = s.find('def build_optimizers')
print(s[i:i + 2600])
