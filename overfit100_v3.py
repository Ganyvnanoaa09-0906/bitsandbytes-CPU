"""overfit100_v3.py -- the decisive "can this architecture fit at all?" run, orchestrated.

200 rows (100 distinct pictures x 2) x 1024 tokens, batch 2. 2000 steps = 20 epochs. Segmented so the
run can sample 8 images after 100 / 250 / 500 / 1000 / 1500 / 2000 global steps: "has it memorised yet"
becomes a curve instead of a single endpoint. Same model/optimizer/schedule as the grid-32 runs, so the
loss is directly comparable with the long run's logs.

Run: python -u overfit100_v3.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ROOT = r'D:\work\bitsandbytes-CPU'
OUT = os.path.join(ROOT, r'i5build\ar_overfit100_v3')
LOG = os.path.join(ROOT, 'overfit100_v3.log')
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
    f.write('overfit100 v3 start ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
say('goal: 2000 steps = 20 epochs over 200 rows x 1024 tokens · 8 samples at 100/250/500/1000/1500/2000')

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
            '--muon-wd', '0.05', '--adamw-wd', '0.05'] + resume
    seg_log = os.path.join(ROOT, 'ofv3_seg_%d.log' % total)
    t0 = time.time()
    with open(seg_log, 'w', encoding='utf-8') as f:
        rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT)
    say('  train exit %d  (%.1f s)' % (rc, time.time() - t0))

    say('sampling 8 images at step %d' % total)
    sample_args = [sys.executable, '-u', 'sample_ar.py', '--ar', ck, '--vqvae', VQ,
                   '--out', os.path.join(OUT, 'samples_step%d.png' % total),
                   '--n', '8', '--seed', '0', '--threads', '5']
    with open(os.path.join(ROOT, 'ofv3_samp_%d.log' % total), 'w', encoding='utf-8') as f:
        src = subprocess.call(sample_args, stdout=f, stderr=subprocess.STDOUT)
    say('  sample exit %d' % src)

    # fold the last validation lines into the main log
    try:
        vals = [ln.strip() for ln in open(seg_log, encoding='utf-8', errors='ignore') if 'val=' in ln][-2:]
        for v in vals:
            with open(LOG, 'a', encoding='utf-8') as f:
                f.write('    ' + v + '\n')
    except OSError:
        pass

say('all segments done ' + time.strftime('%Y-%m-%d %H:%M:%S'))
