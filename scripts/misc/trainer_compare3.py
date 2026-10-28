"""Read the three curves, handling the encoding difference that broke the previous attempt.

trainer_muon3.log was written by Start-Process -RedirectStandardOutput, which is UTF-8.
trainer_muon3_lrfix.log was written by PowerShell's *> redirection, which uses the console
codepage -- so the Chinese text is mojibake and a naive UTF-8 read can lose the ASCII parts
with it. The same mistake already broke the unattended verdict writer once (7.31.4.9), so this
tries encodings in order and reports which one worked rather than assuming.

Then prints the three-way comparison, which is the point:

    AdamW                    i5build/sgdrA.log
    Muon+NS=3, cosine LR     trainer_muon3.log        lost by ~0.5 everywhere
    Muon+NS=3, pinned LR     trainer_muon3_lrfix.log  the fix
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

ROOT = r'D:\work\bitsandbytes-CPU'
PAT = re.compile(r'step (\d+)\s+train=[\d.]+\s+val=([\d.]+)')


def curve(path):
    raw = open(path, 'rb').read()
    for enc in ('utf-8', 'utf-16', 'cp936', 'gbk', 'latin-1'):
        try:
            txt = raw.decode(enc)
        except Exception:
            continue
        vals = [(int(m.group(1)), float(m.group(2))) for m in PAT.finditer(txt)]
        if vals:
            return vals, enc
    return [], None


A = os.path.join(ROOT, 'i5build', 'sgdrA.log')
B = os.path.join(ROOT, 'trainer_muon3.log')
C = os.path.join(ROOT, 'trainer_muon3_lrfix.log')

va, ea = curve(A)
vb, eb = curve(B)
vc, ec = curve(C)
print('  AdamW           %d 点（编码 %s）' % (len(va), ea))
print('  Muon 退火 lr    %d 点（编码 %s）' % (len(vb), eb))
print('  Muon 固定 lr    %d 点（编码 %s）' % (len(vc), ec))
print()

if not vc:
    print('  固定 lr 那次的验证点仍读不出来 —— 需要直接看文件。')
    sys.exit(1)

print('  步数     AdamW      Muon退火    Muon固定    固定 vs AdamW')
for i in range(max(len(va), len(vb), len(vc))):
    s = (vc[i][0] if i < len(vc) else (va[i][0] if i < len(va) else '—'))
    la = va[i][1] if i < len(va) else None
    lb = vb[i][1] if i < len(vb) else None
    lc = vc[i][1] if i < len(vc) else None
    d = '%+.4f' % (lc - la) if (la is not None and lc is not None) else '—'
    print('  %-8s %9s %11s %11s %13s'
          % (s, '%.4f' % la if la is not None else '—',
             '%.4f' % lb if lb is not None else '—',
             '%.4f' % lc if lc is not None else '—', d))

print()
print('  到同一 val 的步数:')
print('    %-8s %9s %11s %11s' % ('目标', 'AdamW', 'Muon退火', 'Muon固定'))
def first(v, t):
    for s, x in v:
        if x <= t:
            return s
    return None
for t in (6.4, 6.2, 6.0, 5.9, 5.8, 5.7, 5.6, 5.55):
    print('    %-8.2f %9s %11s %11s'
          % (t, first(va, t) or '未达到', first(vb, t) or '未达到', first(vc, t) or '未达到'))

# step cost
def steps(path):
    txt = open(path, 'rb').read().decode('utf-8', errors='replace')
    if 'loss=' not in txt:
        for enc in ('utf-16', 'cp936'):
            try:
                txt = open(path, 'rb').read().decode(enc)
                if 'loss=' in txt:
                    break
            except Exception:
                continue
    return [float(m.group(2)) for m in re.finditer(r'\d+/\d+ loss=[\d.]+ ([\d.]+)s', txt)]

import statistics  # noqa: E402
ta, tb, tc = steps(A), steps(B), steps(C)
print()
for name, ts in (('AdamW', ta), ('Muon 退火', tb), ('Muon 固定', tc)):
    if ts:
        print('  %s 每步中位: %.3f s' % (name, statistics.median(ts)))
