"""Declare + bind the 4-bit optimizer so it can actually be called.

Two edits, both mechanical copies of the 8-bit lines with the name changed:
  cpu_ops.h          -- declaration next to the 8-bit one
  pythonInterface.cpp -- c... forwarding wrapper next to the 8-bit one (:917)
"""
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'D:\work\bnb-4bitopt\bitsandbytes\csrc'

# ---- 1) cpu_ops.h declaration ----
P = ROOT + r'\cpu_ops.h'
s = open(P, encoding='utf-8').read()
if 'optimizer_update_4bit_blockwise_cpu' in s:
    print('  = cpu_ops.h 已有声明')
else:
    m = re.search(r'(void optimizer_update_8bit_blockwise_cpu\(.*?\);\n)', s, re.S)
    if not m:
        print('  x cpu_ops.h 找不到 8-bit 声明'); sys.exit(1)
    decl = m.group(1).replace('optimizer_update_8bit_blockwise_cpu',
                              'optimizer_update_4bit_blockwise_cpu')
    note = ('\n// 4-bit blockwise: 状态是一字节两个 4-bit 码（高半字节 = 偶数下标）。\n'
            '// ademamix 第三状态接在 state1 之后，偏移 (n+1)/2 字节。\n')
    s = s.replace(m.group(1), m.group(1) + note + decl, 1)
    open(P, 'w', encoding='utf-8', newline='').write(s)
    print('  + cpu_ops.h 已加声明')

# ---- 2) pythonInterface.cpp binding ----
Q = ROOT + r'\pythonInterface.cpp'
q = open(Q, encoding='utf-8').read()
if 'coptimizer_update_4bit_blockwise_cpu' in q:
    print('  = pythonInterface.cpp 已有绑定')
else:
    m = re.search(r'(void coptimizer_update_8bit_blockwise_cpu\(.*?\n\}\n)', q, re.S)
    if not m:
        print('  x pythonInterface.cpp 找不到 8-bit 绑定'); sys.exit(1)
    wrap = m.group(1).replace('coptimizer_update_8bit_blockwise_cpu',
                              'coptimizer_update_4bit_blockwise_cpu') \
                     .replace('optimizer_update_8bit_blockwise_cpu',
                              'optimizer_update_4bit_blockwise_cpu')
    header = ('\n// 4-bit blockwise optimizer step. Same contract as the 8-bit wrapper;\n'
              '// state1/state2 are ceil(n/2) bytes each (two 4-bit codes per byte,\n'
              '// high nibble = even index).\n')
    q = q.replace(m.group(1), m.group(1) + header + wrap, 1)
    open(Q, 'w', encoding='utf-8', newline='').write(q)
    print('  + pythonInterface.cpp 已加绑定')

# ---- 3) where does Python declare these c... symbols? ----
import os
hits = []
for dirpath, dirnames, filenames in os.walk(r'D:\work\bnb-4bitopt\bitsandbytes'):
    if '__pycache__' in dirpath or 'build' in dirpath:
        continue
    for fn in filenames:
        if not fn.endswith('.py'):
            continue
        fp = os.path.join(dirpath, fn)
        try:
            t = open(fp, encoding='utf-8', errors='replace').read()
        except Exception:
            continue
        if 'optimizer_update_8bit_blockwise' in t or 'optimizer_update_32bit' in t:
            hits.append((fp, t.count('optimizer_update_')))
print('\n  Python 侧声明这些符号的文件:')
for fp, n in hits[:6]:
    print('    %s  (%d 处)' % (fp.replace(r'D:\work\bnb-4bitopt\\', ''), n))
