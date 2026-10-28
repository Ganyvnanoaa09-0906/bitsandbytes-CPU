"""overfit100_bos.py -- the §7.111 confound removed: the same 20-epoch overfit run, with BOS.

The 2000-step overfit-100 run concluded "teacher forcing reconstructs recognisable pictures, free
sampling still collapses", but it used `--use_bos False`: the model never saw a BOS, so free
generation started from token ID 0 (the code calls this the root cause of cold-start collapse). That
run therefore cannot say how much of the failure is the BOS bug and how much is exposure bias.

This is the same experiment with `--use_bos` on: 200 rows (100 pictures x2) x 1024 tokens, batch 2,
2000 steps = 20 epochs, identical model/optimizer/schedule, segmented for samples at
100/250/500/1000/1500/2000.

Run: python -u overfit100_bos.py
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ROOT = r'D:\work\bitsandbytes-CPU'
OUT = os.path.join(ROOT, r'i5build\ar_overfit100_bos')
LOG = os.path.join(ROOT, 'overfit100_bos.log')
TOK = 'tokens_overfit100.pt'
VQ = r'i5build\vqgan32\ckpt_last.pt'
SEGMENTS = [100, 150, 250, 500, 500, 500]   # cumulative: 100, 250, 500, 1000, 1500, 2000
os.makedirs(OUT, exist_ok=True)
os.chdir(ROOT)


def say(msg):
    line = time.strftime('%H:%M:%S') + '  ' + msg
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


with open(LOG, 'w', encoding='utf-8') as f:
    f.write('overfit100 BOS start ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
say('goal: 2000 steps = 20 epochs with --use_bos (the 7.111 run used use_bos=False)')

total = 0
for per in SEGMENTS:
    total += per
    ck = os.path.join(OUT, 'ckpt_last.pt')
    resume = ['--resume', ck] if os.path.exists(ck) else []
    say('segment -> global step %d (+%d)  (%s)' % (total, per, 'resuming' if resume else 'fresh'))
    args = [sys.executable, '-u', 'train_ar_v2.py',
            '--tokens', TOK, '--vqvae', VQ, '--out', OUT,
            '--stages', '1024:%d' % total, '--batch', '1024:2',
            '--lr', '1e-4', '--warmup', '100', '--lr-min', '1e-5',
            '--n_layer', '10', '--n_cluster', '32',
            '--head-mode', 'flat', '--mask-mode', 'causal',
            '--threads', '5', '--ckpt_every', '125',
            '--loop_times', '1', '--loop_L', '1', '--loop_L_max', '3',
            '--cond-mode', 'none', '--val_frac', '0.02', '--val_every', '50',
            '--optimizer', 'muon', '--ns-iters', '3',
            '--muon-wd', '0.05', '--adamw-wd', '0.05',
            '--use_bos'] + resume
    seg_log = os.path.join(ROOT, 'ofbos_seg_%d.log' % total)
    t0 = time.time()
    with open(seg_log, 'w', encoding='utf-8') as f:
        rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT)
    say('  train exit %d  (%.1f s)' % (rc, time.time() - t0))
    try:
        vals = [ln.strip() for ln in open(seg_log, encoding='utf-8', errors='ignore') if 'val=' in ln][-2:]
        for v in vals:
            with open(LOG, 'a', encoding='utf-8') as f:
                f.write('    ' + v + '\n')
    except OSError:
        pass

    say('evaluating memorisation at step %d' % total)
    eval_args = [sys.executable, '-u', 'eval_memorize.py',
                 '--ar', ck, '--vqvae', VQ, '--tokens', TOK, '--n', '8',
                 '--out-prefix', os.path.join(OUT, 'memorize_step%d' % total)]
    with open(os.path.join(ROOT, 'ofbos_eval_%d.log' % total), 'w', encoding='utf-8') as f:
        erc = subprocess.call(eval_args, stdout=f, stderr=subprocess.STDOUT)
    say('  eval exit %d' % erc)
    try:
        for ln in open(os.path.join(ROOT, 'ofbos_eval_%d.log' % total), encoding='utf-8', errors='ignore'):
            if ln.startswith('[') or 'wrote' in ln:
                with open(LOG, 'a', encoding='utf-8') as f:
                    f.write('    ' + ln.strip() + '\n')
    except OSError:
        pass

    say('sampling 8 free images at step %d' % total)
    sample_args = [sys.executable, '-u', 'sample_ar.py', '--ar', ck, '--vqvae', VQ,
                   '--out', os.path.join(OUT, 'samples_step%d.png' % total),
                   '--n', '8', '--seed', '0', '--threads', '5']
    with open(os.path.join(ROOT, 'ofbos_samp_%d.log' % total), 'w', encoding='utf-8') as f:
        src = subprocess.call(sample_args, stdout=f, stderr=subprocess.STDOUT)
    say('  sample exit %d' % src)

say('all segments done ' + time.strftime('%Y-%m-%d %H:%M:%S'))
