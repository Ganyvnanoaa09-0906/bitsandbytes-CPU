# -*- coding: utf-8 -*-
import io, re
s = io.open(r'D:\work\bitsandbytes-CPU\train_ar_v2.py', encoding='utf-8').read()
i = s.find('_OPTS')
print(s[max(0, i - 2500):i + 300])
