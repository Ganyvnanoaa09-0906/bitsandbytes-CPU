"""run_sgdr_ab.py -- drive the two SGDR arms sequentially and compare validation loss.

The user's suggestion, and the project's own curve motivates it. Validation loss on the
3000-step run improved by -0.256, -0.105, -0.051, -0.028, -0.014 over successive 500-step
windows -- halving each time. That is what a single cosine anneal looks like: as lr falls
the steps shorten and the model settles into whichever basin it happened to reach. SGDR
restarts the cosine so lr returns to its peak, giving the model a chance to leave that
basin.

Judged on validation loss, not training loss: the per-step training loss has a standard
deviation around 0.58, so anything but a validation curve is unreadable. That is why both
arms use --val_frac and --val_every, which the earlier 3000-step run did not.

Two arms, identical except for the schedule:

  A  --lr-restarts 1    the current single anneal
  B  --lr-restarts 2    two cycles over the same span, so lr jumps back at step 750

1500 steps each at batch 2, T=1024, which is the configuration the 3.671 s/step figure came
from, so roughly 1.5 h per arm. The prediction to test: if restarts help, B should show a
second large drop in validation loss after step 750, of the order of the -0.25 seen in the
first 500 steps, rather than continuing the halving decay. Stated before running, so it
cannot be reinterpreted afterwards.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import json
import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

os.chdir(r'D:\work\bitsandbytes-CPU')

TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
VQVAE = r'i5build\vqgan32\ckpt_last.pt'

COMMON = [
    'python', '-u', 'train_ar_v2.py',
    '--tokens', TOKENS, '--vqvae', VQVAE,
    '--stages', '1024:1500', '--batch', '1024:2',
    '--lr', '1e-4', '--warmup', '100', '--lr-min', '1e-5',
    '--n_layer', '10', '--n_cluster', '32',
    '--head-mode', 'flat', '--mask-mode', 'causal',
    '--threads', '6', '--ckpt_every', '1500',
    '--loop_times', '1', '--loop_L', '1', '--loop_L_max', '3',
    '--cond-mode', 'none',
    # validation loss is the whole point of this comparison
    '--val_frac', '0.02', '--val_every', '250',
]

ARMS = [('sgdrA', ['--lr-restarts', '1'], '单次余弦退火（现状）'),
        ('sgdrB', ['--lr-restarts', '2'], '两周期（750 步一个，lr 会跳回峰值）')]

print('SGDR A/B: 1500 步 × 2 臂，batch 2，T=1024，验证 loss 每 250 步')
print('判据（先声明）: 若 B 在 750 步之后出现【第二次大幅下降】（−0.2 量级），')
print('                而不是延续 −0.25 → −0.10 → −0.05 的减半衰减 ⇒ SGDR 有效。')
print()

for tag, extra, desc in ARMS:
    out = os.path.join('i5build', tag)
    log = os.path.join('i5build', tag + '.log')
    os.makedirs(out, exist_ok=True)
    if os.path.exists(log):
        os.remove(log)
    cmd = COMMON + ['--out', out] + extra
    print('=== 臂 %s: %s ===' % (tag, desc))
    print('  ' + ' '.join(cmd[2:]))
    t0 = time.perf_counter()
    with open(log, 'w', encoding='utf-8') as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    print('  退出码 %d，用时 %.1f 分钟' % (p.returncode, dt / 60))
    # report the validation entries
    try:
        with open(log, encoding='utf-8', errors='replace') as fh:
            vals = [ln.strip() for ln in fh if 'val' in ln.lower() and 'loss' in ln.lower()]
        for v in vals[-8:]:
            print('    ' + v[:110])
    except Exception as e:
        print('    读取日志失败: %s' % e)
    print()

print('=== 两臂的验证曲线对比 ===')
res = {}
for tag, _, desc in ARMS:
    log = os.path.join('i5build', tag + '.log')
    vals = []
    try:
        with open(log, encoding='utf-8', errors='replace') as fh:
            for ln in fh:
                if 'val' in ln.lower() and 'loss' in ln.lower():
                    import re
                    m = re.findall(r'[-+]?\d*\.\d+', ln)
                    if m:
                        vals.append((int(re.findall(r'\d+', ln)[0]) if re.findall(r'\d+', ln) else 0,
                                     float(m[-1])))
    except Exception:
        pass
    res[tag] = vals
    print('  %s: %d 个验证点' % (tag, len(vals)))
    for s, v in vals:
        print('    步 %-6d loss %.4f' % (s, v))
