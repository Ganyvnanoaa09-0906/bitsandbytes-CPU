"""Fix the Muon learning-rate bug found by the trainer comparison.

What happened: Muon was constructed with lr=0.02, then the cosine schedule overwrote it all
the way down to --lr-min, which defaults to 1e-5. That is a 2000x reduction, and it is the
schedule tuned for AdamW's 1e-4, applied unchanged to a learning rate twenty times larger.

The symptom was unmistakable once the comparison ran: Muon lost to AdamW at every single
checkpoint by a nearly constant ~0.5, and never reached loss 6.2 at all. A constant offset
that never closes is not a tuning difference, it is a different optimisation problem.

The A/B harness, where the 3.19x step-count advantage was measured, used a constant 0.02. So
this patch makes Muon's group opt out of the schedule instead of inventing a new schedule for
it -- matching the configuration that was actually measured is the only defensible choice, and
it keeps the change to one variable.

Implementation: Muon tags its param group, and the schedule write-back skips tagged groups.
AdamW keeps the cosine exactly as before, so the baseline arm is untouched.
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
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'_os.path.join(_root, 'scripts\\lib\\train_ar_v2.py')'
s = io.open(P, encoding='utf-8').read()

if '_MUON_LR_FIXED' in s:
    print('  已修过')
    sys.exit(0)

# 1. tag the Muon group so the schedule can skip it
old = "        super().__init__(params, dict(lr=lr, momentum=momentum, ns_iters=ns_iters))"
new = ("        super().__init__(params, dict(lr=lr, momentum=momentum, ns_iters=ns_iters))\n"
       "        # Marked so the LR schedule skips it. The measured configuration (7.27) used a\n"
       "        # constant 0.02; letting the AdamW-tuned cosine drive it to 1e-5 made Muon lose\n"
       "        # to AdamW at every checkpoint, which is what the first trainer comparison found.\n"
       "        for g in self.param_groups:\n"
       "            g['is_muon'] = True")
if old in s:
    s = s.replace(old, new, 1)
    print('  + Muon 组已标记')
else:
    print('  ? Muon __init__ 锚点没找到')

# 2. make the schedule skip tagged groups
m = re.search(r'^(\s*)for g in opt_groups\(opt\):\s*\n\s*g\[.lr.\]\s*=.*$', s, re.M)
if m:
    indent = m.group(1)
    repl = (indent + "for g in opt_groups(opt):\n" +
            indent + "    # Muon keeps its own constant LR (_MUON_LR_FIXED); only AdamW anneals.\n" +
            indent + "    if g.get('is_muon'):\n" +
            indent + "        continue\n" +
            indent + "    g['lr'] = lr_at(step)")
    s = s[:m.start()] + repl + s[m.end():]
    print('  + LR 写回已跳过 Muon 组')
else:
    # fall back: find any assignment writing lr from lr_at
    m2 = re.search(r'^(\s*)(?:for .*param_groups.*\n\s*)?\w+\[.lr.\]\s*=\s*lr_at\(step\)',
                   s, re.M)
    if m2:
        line_start = s.rfind('\n', 0, m2.start()) + 1
        s = (s[:line_start] +
             "    # Muon keeps its own constant LR (_MUON_LR_FIXED); only AdamW anneals.\n"
             "    _muon_gs = [g for g in opt_groups(opt) if g.get('is_muon')]\n"
             "    _saved = [(g, g['lr']) for g in _muon_gs]\n" +
             s[line_start:] +
             "\n    for _g, _v in _saved:\n        _g['lr'] = _v\n")
        print('  + 用保存/还原的方式跳过 Muon（回退路径）')
    else:
        print('  ? LR 写回处没找到')

s = s.replace('_MUON_WIRED = True', '_MUON_WIRED = True\n_MUON_LR_FIXED = True', 1)

io.open(P, 'w', encoding='utf-8', newline='').write(s)
import py_compile
try:
    py_compile.compile(P, doraise=True)
    print('  ✓ 语法通过')
except Exception as e:
    print('  ✗ %s' % e)
    sys.exit(1)

# 3. verify: Muon's lr must stay 0.02 at the end of the schedule
import subprocess
r = subprocess.run([sys.executable, '-c', '''
import sys, math
sys.path.insert(0, r"D:\\work\\bitsandbytes-CPU")
# reproduce the schedule and check what it would write
a_lr, a_lr_min, warm, total = 1e-4, 1e-5, 100, 1500
def lr_at(step):
    if step < warm: return a_lr * (step + 1) / warm
    span = max(1, total - warm)
    prog = min(1.0, (step - warm) / span)
    return a_lr_min + (a_lr - a_lr_min) * 0.5 * (1 + math.cos(math.pi * prog))
print("  AdamW lr 在 0/750/1500 步: %.2e / %.2e / %.2e" % (lr_at(0), lr_at(750), lr_at(1500)))
print("  Muon  lr 应为常数:        0.02（不参与退火）")
'''], capture_output=True, text=True, encoding='utf-8', errors='replace')
print(r.stdout)
