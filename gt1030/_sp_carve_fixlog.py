# -*- coding: utf-8 -*-
"""修上一步把 logmsg 参数顺序插错的语句。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_carve.c'
L = io.open(p, encoding='utf-8').read().split('\n')
i = next(k for k, ln in enumerate(L) if 'logmsg(L"  [%ls] %ls type=%hs' in ln)
# 原来的语句跨 3 行（logmsg 行 + 参数行 + 我误插的 tag 行）
j = i
while ';' not in L[j]:
    j += 1
new = ['    logmsg(L"  [%ls] %ls type=%hs len=%llu fileStart=%llu\\n",',
       '           hitTail ? L"ok" : L"partial", path, ext,',
       '           (unsigned long long)size, (unsigned long long)fileStart);']
L[i:j + 1] = new
io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('logmsg 语句已修好')
