"""H3 round 2, step 4: add --ns-iters to the Muon path, then run the short A/B.

Step 3 found a real warning that has to be tested rather than argued about. Shortening NS
from 5 iterations to 3 saves 763 ms of a 3952 ms step -- a 1.24x ceiling -- but it also
raises the condition number of the 512x512 updates from 27 to 320, a factor of 13. Whether
that matters for training is not something the microbenchmark can answer.

So: one variable, NS iteration count. Everything else fixed -- same data, same model, same
seed, same learning rate, same step count, same validation set and frequency. Two arms, 600
steps each, which at the measured 6.4 s per Muon step is about an hour per arm.

600 steps is a screening length, not a verdict. The Muon-vs-AdamW comparison showed the gap
narrowing with length, so a short run can mislead about long-run behaviour; if the arms are
close here, that is reason to run longer rather than reason to conclude.

Adds --ns-iters to train_ar_v2.py's Muon path, defaulting to 5 so the baseline is untouched.
The change is recorded as its own commit so it can be reverted independently.
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import io
import os
import re
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

ROOT = r'D:\work\bitsandbytes-CPU'
os.chdir(ROOT)
P = os.path.join(ROOT, 'train_ar_v2.py')

# ---------------------------------------------------------------- 1. find the Muon path
s = io.open(P, encoding='utf-8').read()
print('  Muon 相关的位置:')
for m in re.finditer(r'^.*(newtonschulz|NewtonSchulz|zeropower|NS_STEPS|ns_steps|muon).*$',
                     s, re.M | re.I):
    line = m.group(0).strip()
    if len(line) < 130:
        print('    L%d: %s' % (s[:m.start()].count('\n') + 1, line[:110]))

# ---------------------------------------------------------------- 2. add --ns-iters
if 'ns_iters' not in s:
    OLD = "    ap.add_argument('--total_tokens', type=int, default=0,"
    NEW = ('''    # ★ H3 实验用：Muon 的 Newton-Schulz 迭代次数。默认 5（原文与基线）。
    # 第二轮微基准：5→3 每步省 763 ms（优化器 2562→1799 ms），但 512×512 更新的条件数
    # 从 27 涨到 320。是否可接受只能在训练里测，见 research/ns_iterations.md。
    ap.add_argument('--ns-iters', type=int, default=5,
                    help='Muon 的 NS 迭代次数（默认 5；H3 实验用 3 或 4）')
''' + OLD)
    if OLD in s:
        s = s.replace(OLD, NEW, 1)
        print('  + --ns-iters 已加')
    else:
        print('  ? 参数锚点没找到')

# ---------------------------------------------------------------- 3. wire it into the NS call
# find the NS function definition and make its default read from the argument, if a global
# is available; otherwise find where it is called with an explicit step count.
m = re.search(r'def (zeropower\w*|newtonschulz\w*)\(([^)]*)\)', s)
if m:
    print('  NS 函数: %s(%s)' % (m.group(1), m.group(2)[:80]))
    # add a module-level override the function reads
    if '_NS_ITERS_OVERRIDE' not in s:
        sig_end = m.end()
        # insert the override right after the def line
        body_start = s.find('\n', sig_end) + 1
        inj = ('    global _NS_ITERS_OVERRIDE\n'
               '    if _NS_ITERS_OVERRIDE is not None:\n'
               '        steps = _NS_ITERS_OVERRIDE\n')
        s = s[:body_start] + inj + s[body_start:]
        s = s.replace('def %s(' % m.group(1),
                      '_NS_ITERS_OVERRIDE = None\n\n\ndef %s(' % m.group(1), 1)
        print('  + NS 函数已加全局覆盖钩子')

    # set the override from the argument, near where the optimiser is built
    for anchor in ("muon_p", "muon_params", "Muon("):
        k = s.find(anchor)
        if k > 0:
            line_start = s.rfind('\n', 0, k) + 1
            s = (s[:line_start] +
                 '    # ★ H3：把 --ns-iters 传给 NS 实现（默认 5 ⇒ 基线不变）\n'
                 '    _NS_ITERS_OVERRIDE = int(getattr(a, "ns_iters", 5) or 5)\n' +
                 s[line_start:])
            print('  + 覆盖值已接在 %r 之前（L%d）'
                  % (anchor, s[:line_start].count('\n') + 1))
            break

io.open(P, 'w', encoding='utf-8', newline='').write(s)
import py_compile
try:
    py_compile.compile(P, doraise=True)
    print('  ✓ 语法通过')
except Exception as e:
    print('  ✗ 语法错误: %s' % e)
    sys.exit(1)

# ---------------------------------------------------------------- 4. smoke both arms
print()
print('  冒烟：两臂各 2 步，确认 ns_iters 被接受')
TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
VQ = r'i5build\vqgan32\ckpt_last.pt'
for it in (5, 3):
    cmd = ['python', '-u', 'train_ar_v2.py', '--tokens', TOK, '--vqvae', VQ,
           '--out', 'i5build/_ns_smoke', '--stages', '1024:2', '--batch', '1024:2',
           '--lr', '1e-4', '--warmup', '1', '--lr-min', '1e-5', '--n_layer', '10',
           '--n_cluster', '32', '--head-mode', 'flat', '--mask-mode', 'causal',
           '--threads', '5', '--ckpt_every', '100', '--loop_times', '1', '--loop_L', '1',
           '--loop_L_max', '3', '--cond-mode', 'none', '--val_frac', '0.02',
           '--val_every', '5', '--ns-iters', str(it), '--optimizer', 'muon']
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=1800)
    tail = [ln for ln in (r.stdout or '').split('\n') if 'loss=' in ln or 'Error' in ln]
    print('    ns-iters=%d → 退出码 %d  %s' % (it, r.returncode, tail[-1][:80] if tail else ''))
    if r.returncode != 0:
        print((r.stdout or '')[-800:])
        print((r.stderr or '')[-400:])
