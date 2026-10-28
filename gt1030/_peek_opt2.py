# -*- coding: utf-8 -*-
import io, re
s = io.open(r'D:\work\bitsandbytes-CPU\train_ar_v2.py', encoding='utf-8').read()
# find the actual optimizer object creation
for m in re.finditer(r'\n\s*opt\w*\s*=\s*[^\n]+', s):
    print(m.group(0).strip())
print('--- betas ---')
for m in re.finditer(r'betas\s*=\s*\([^)]*\)', s):
    print(m.group(0))
print('--- AdamW( ---')
for m in re.finditer(r'AdamW\(', s):
    a = max(0, m.start() - 300)
    print('...', s[a:m.end() + 260].replace('\n', ' | ')[-460:])
    print()
