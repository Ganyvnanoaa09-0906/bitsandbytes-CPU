"""overfit_single.py -- can the 30.5M AR model memorise TWO sequences at all?

The overfit-100 run (report 7.111) left a gap: after 20 epochs the teacher-forced reconstruction of
the training pictures was recognisable, but nothing was memorised to CE ~ 0, so "can it fit at all?"
was still open. The cheapest decisive version is this one: two rows (one picture x2), 300 steps,
same model/optimizer/threads as the grid-32 runs. If CE cannot approach 0 here, the recipe or the
architecture is the problem, not the amount of data.

Runs: train 300 steps (segmented 150+150), then eval_memorize (CE, greedy reconstruction, free
samples), so the answer is a curve plus a picture rather than one number.

Run: python -u overfit_single.py
"""
from __future__ import annotations
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

import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ROOT = r'D:\work\bitsandbytes-CPU'
OUT = os.path.join(ROOT, r'i5build\of_single')
LOG = os.path.join(ROOT, 'overfit_single.log')
TOK = 'tokens_single2.pt'
VQ = r'i5build\vqgan32\ckpt_last.pt'
SEGMENTS = [150, 150]   # cumulative 150, 300
os.makedirs(OUT, exist_ok=True)
os.chdir(ROOT)


def say(msg):
    line = time.strftime('%H:%M:%S') + '  ' + msg
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


with open(LOG, 'w', encoding='utf-8') as f:
    f.write('overfit_single start ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
say('goal: 2 rows x 1024 tokens, 300 steps, batch 2 -- can CE reach ~0?')

total = 0
for per in SEGMENTS:
    total += per
    ck = os.path.join(OUT, 'ckpt_last.pt')
    resume = ['--resume', ck] if os.path.exists(ck) else []
    say('segment -> %d steps (+%d)  (%s)' % (total, per, 'resuming' if resume else 'fresh'))
    args = [sys.executable, '-u', 'train_ar_v2.py',
            '--tokens', TOK, '--vqvae', VQ, '--out', OUT,
            '--stages', '1024:%d' % total, '--batch', '1024:2',
            '--lr', '1e-4', '--warmup', '50', '--lr-min', '1e-5',
            '--n_layer', '10', '--n_cluster', '32',
            '--head-mode', 'flat', '--mask-mode', 'causal',
            '--threads', '5', '--ckpt_every', '75',
            '--loop_times', '1', '--loop_L', '1', '--loop_L_max', '3',
            '--cond-mode', 'none', '--val_frac', '0.0', '--val_every', '1000',
            '--optimizer', 'muon', '--ns-iters', '3',
            '--muon-wd', '0.05', '--adamw-wd', '0.05'] + resume
    seg_log = os.path.join(ROOT, 'ofsingle_seg_%d.log' % total)
    t0 = time.time()
    with open(seg_log, 'w', encoding='utf-8') as f:
        rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT)
    say('  train exit %d  (%.1f s)' % (rc, time.time() - t0))
    try:
        lines = [ln.strip() for ln in open(seg_log, encoding='utf-8', errors='ignore') if 'loss=' in ln]
        for ln in lines[-3:]:
            with open(LOG, 'a', encoding='utf-8') as f:
                f.write('    ' + ln + '\n')
    except OSError:
        pass

say('evaluating memorisation (CE / greedy reconstruction / free samples)')
eval_args = [sys.executable, '-u', 'eval_memorize.py',
             '--ar', os.path.join(OUT, 'ckpt_last.pt'), '--vqvae', VQ,
             '--tokens', TOK, '--n', '2',
             '--out-prefix', os.path.join(OUT, 'memorize')]
with open(os.path.join(ROOT, 'ofsingle_eval.log'), 'w', encoding='utf-8') as f:
    erc = subprocess.call(eval_args, stdout=f, stderr=subprocess.STDOUT)
say('  eval exit %d' % erc)
try:
    for ln in open(os.path.join(ROOT, 'ofsingle_eval.log'), encoding='utf-8', errors='ignore'):
        if ln.startswith('[') or 'wrote' in ln:
            with open(LOG, 'a', encoding='utf-8') as f:
                f.write('    ' + ln.strip() + '\n')
except OSError:
    pass

say('sampling 8 free images')
sample_args = [sys.executable, '-u', 'sample_ar.py',
               '--ar', os.path.join(OUT, 'ckpt_last.pt'), '--vqvae', VQ,
               '--out', os.path.join(OUT, 'samples_free.png'),
               '--n', '8', '--seed', '0', '--threads', '5']
with open(os.path.join(ROOT, 'ofsingle_sample.log'), 'w', encoding='utf-8') as f:
    src = subprocess.call(sample_args, stdout=f, stderr=subprocess.STDOUT)
say('  sample exit %d' % src)
say('done ' + time.strftime('%Y-%m-%d %H:%M:%S'))
