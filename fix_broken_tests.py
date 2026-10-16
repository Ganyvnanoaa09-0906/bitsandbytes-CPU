"""Repair the two tests my checkpoint cleanup broke, and make them robust against the next one.

test_adamw4bit_real.py and test_4bit_memory_payoff.py both hardcode

    CK = r'D:\\work\\bitsandbytes-CPU\\i5build\\loop_ar3\\ckpt.pt\\ckpt_last.pt'

and that checkpoint no longer exists. I deleted it on 10-15 while clearing 141 intermediate
checkpoints to free 13.7 GB. The cleanup explicitly kept every checkpoint the REPORT referenced; what
it did not check was which ones the TEST SUITE referenced. That is the gap.

So the fix is not to point them at another single file, which the next cleanup would break again. It is
to try a list of candidates and take the first that loads, so the tests survive housekeeping. The list
is ordered with the 32x32 AR models first because that is the architecture these tests want: 30.5M
parameters, grid 32, causal, which is what loop_ar3 was as well.
"""
from __future__ import annotations

import io
import sys

sys.stdout.reconfigure(encoding='utf-8')

OLD = "CK = r'D:\\work\\bitsandbytes-CPU\\i5build\\loop_ar3\\ckpt.pt\\ckpt_last.pt'"

NEW = '''# ★ 不再写死单个 checkpoint。10-15 的磁盘清理删掉了原来这里引用的
#   loop_ar3/ckpt.pt/ckpt_last.pt —— 那次清理【保留了报告引用过的所有 checkpoint】，
#   但没检查【测试引用了哪些】。所以改成候选列表：挑第一个能加载的，
#   这样下一次清理不会再把这两个测试弄坏。
#   顺序:32x32 的 AR 模型在前(30.5M / grid 32 / causal),也就是 loop_ar3 的形状。
CK_CANDIDATES = [
    r'D:\\work\\bitsandbytes-CPU\\i5build\\ar32_run1\\ckpt_last.pt',
    r'D:\\work\\bitsandbytes-CPU\\i5build\\ar_wd\\ckpt_last.pt',
    r'D:\\work\\bitsandbytes-CPU\\i5build\\ar_long\\ckpt_last.pt',
    r'D:\\work\\bitsandbytes-CPU\\i5build\\ar_muon3_3000\\ckpt_last.pt',
    r'D:\\work\\bitsandbytes-CPU\\i5build\\ar_g16\\ckpt_last.pt',
]


def _pick_ckpt():
    import os
    for p in CK_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        'no candidate checkpoint exists; tried:\\n  ' + '\\n  '.join(CK_CANDIDATES))


CK = _pick_ckpt()'''

for path in (r'D:\work\bnb-4bitopt\test_adamw4bit_real.py',
             r'D:\work\bnb-4bitopt\test_4bit_memory_payoff.py'):
    s = io.open(path, encoding='utf-8').read()
    if OLD not in s:
        if 'CK_CANDIDATES' in s:
            print('  已修过: %s' % path.split('\\')[-1])
            continue
        print('  ? 找不到锚点: %s' % path.split('\\')[-1])
        continue
    s = s.replace(OLD, NEW, 1)
    io.open(path, 'w', encoding='utf-8', newline='').write(s)
    import py_compile
    py_compile.compile(path, doraise=True)
    print('  ✓ 已修并语法通过: %s' % path.split('\\')[-1])
