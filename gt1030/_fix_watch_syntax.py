# -*- coding: utf-8 -*-
"""修 _watch_runb.py 里那行语法错误（我加 % TAG 时把隐式字符串拼接破坏了）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\_watch_runb.py'
s = io.open(p, encoding='utf-8').read()
bad = "            f.write('# %s 进度（每 2.5k 步采样 + token 指标）\\n\\n' % TAG\n"
if bad in s:
    s = s.replace(bad, "            f.write('# ' + TAG + ' 进度（每 2.5k 步采样 + token 指标）\\n\\n'\n")
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
    print('fixed')
else:
    # 兜底：找到那行并整行重写
    lines = s.splitlines()
    for i, ln in enumerate(lines):
        if '2.5k' in ln and 'f.write' in ln:
            lines[i] = "            f.write('# ' + TAG + ' 进度（每 2.5k 步采样 + token 指标）\\n\\n'"
            print('fixed at line', i + 1)
            break
    io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
import py_compile
py_compile.compile(p, doraise=True)
print('compile ok')
