"""Export AdamW4bit from bitsandbytes.optim so a pip user can reach it.

Kept as a file rather than an inline python -c: the inline form has now broken
several times tonight on nested quotes (an apostrophe inside a single-quoted
string this time). Writing the file and running it is the rule; follow it.
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\bnb-4bitopt\bitsandbytes\bitsandbytes\optim\__init__.py'
s = open(P, encoding='utf-8').read()
if 'adamw4bit' in s:
    print('  已经导出过，未改动')
else:
    if not s.endswith('\n'):
        s += '\n'
    s += (
        '\n'
        '# 4-bit blockwise AdamW for CPU (this fork kernel).\n'
        '# State costs 1.031 bytes/param against 8 for fp32; report section 7.16.\n'
        'from .adamw4bit import AdamW4bit  # noqa: F401\n'
    )
    open(P, 'w', encoding='utf-8', newline='').write(s)
    print('  + optim/__init__.py 已导出 AdamW4bit')

# sanity: the module must import from inside the package layout
sys.path.insert(0, r'D:\work\bnb-4bitopt\bitsandbytes')
try:
    from bitsandbytes.optim.adamw4bit import AdamW4bit, _dll_path
    print('  + import 成功，DLL =', _dll_path)
except Exception as e:
    print('  x import 失败:', type(e).__name__, e)
