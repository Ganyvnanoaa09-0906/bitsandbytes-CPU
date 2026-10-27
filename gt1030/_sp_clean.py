# -*- coding: utf-8 -*-
"""正常结束时删除 <目标>.progress（补上 commit 里承诺、代码漏掉的一步）。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_mirror.c'
L = io.open(p, encoding='utf-8').read().split('\n')
i = next(k for k, ln in enumerate(L) if ln.strip() == 'free(buf);')
L[i:i] = ['    _wremove(progPath);   /* 正常完成：断点文件作废 */',
          '    wprintf(L"[*] 断点文件已清除\\n");']
io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('added _wremove')
