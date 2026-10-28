"""Fix the indentation my previous patch got wrong.

The function body is at 4 spaces. My replacement strings assumed 8, so the new
--levels argument landed at column 0 and the CH_BY_LEVELS block at 8. Read the
exact lines and rewrite them, rather than reconstructing from memory again.
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\bitsandbytes-CPU\small_vqvae.py'
s = io.open(P, encoding='utf-8').read()

# 1) re-indent the --levels argument block to 4 spaces
s = s.replace(
    "ap.add_argument('--n_codes', type=int, default=1024)\n"
    "ap.add_argument('--levels', type=int, default=4,\n"
    "                help='ch 的长度 = log2(下采样倍率): 4 -> 16x -> 16x16 网格, '\n"
    "                     '3 -> 8x -> 32x32 网格（同 size 下 token 数 4 倍，'\n"
    "                     'tokenizer 成本约 4 倍降为 1 倍）')",
    "    ap.add_argument('--n_codes', type=int, default=1024)\n"
    "    ap.add_argument('--levels', type=int, default=4,\n"
    "                    help='ch 的长度 = log2(下采样倍率): 4 -> 16x -> 16x16 网格, '\n"
    "                         '3 -> 8x -> 32x32 网格')",
    1)

# 2) de-indent the CH_BY_LEVELS block from 8 to 4
s = re.sub(
    r"\n    CH_BY_LEVELS = \{([^}]*)\}\n(        [^\n]*\n)+",
    lambda m: ("\n    CH_BY_LEVELS = {%s}\n"
               "    _ch = CH_BY_LEVELS.get(a.levels)\n"
               "    if _ch is None:\n"
               "        raise SystemExit('--levels must be 3, 4 or 5; got %r' % a.levels)\n" % m.group(1)),
    s, count=1)

# 3) make sure the constructor uses _ch
s = s.replace('model=VQVAE(embed_dim=a.embed_dim,n_codes=a.n_codes,ch=_ch)',
              'model=VQVAE(embed_dim=a.embed_dim,n_codes=a.n_codes,ch=_ch)', 1)

io.open(P, 'w', encoding='utf-8', newline='').write(s)

import py_compile
try:
    py_compile.compile(P, doraise=True)
    print('  ✓ 语法通过')
except Exception as e:
    print('  ✗ %s' % e)
    # show the offending region so the next attempt is not another guess
    for i, l in enumerate(s.split('\n')[243:256], start=244):
        print('    %d: %s' % (i, l))
