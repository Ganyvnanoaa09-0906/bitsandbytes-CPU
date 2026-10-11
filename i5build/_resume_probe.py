import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
s = open(r'D:\work\bitsandbytes-CPU\train_ar_v2.py', encoding='utf-8').read()
m = re.search(r"add_argument\('--resume'.*?\)\n", s, re.S)
print('参数定义:', (m.group(0).strip() if m else '没找到'))
for m in re.finditer(r'a\.resume', s):
    i = m.start()
    print('---')
    print(s[max(0, i - 200):i + 300])
