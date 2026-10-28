# -*- coding: utf-8 -*-
import io
s = io.open(r'D:\work\bitsandbytes-CPU\train_ar_v2.py', encoding='utf-8').read()
i = s.find('torch.optim.AdamW')
print(s[max(0, i - 900):i + 800])
print('=== --opt flag ===')
j = s.find("'--opt'")
print(s[j - 100:j + 400] if j > 0 else 'no --opt')
