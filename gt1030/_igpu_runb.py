# -*- coding: utf-8 -*-
"""run B：从 ckpt_30000（干净、val 5.1963）续跑 2 万步，配方更稳（lr 2e-5→3e-6、warmup 200、
NaN 守卫已生效），每 2500 步存 ckpt；跑完采样 + 写笔记。

    python gt1030/_igpu_runb.py
"""
from __future__ import annotations
import io
import os
import subprocess
import sys
import time

PY = sys.executable
BASE = r'D:\work\bitsandbytes-CPU'
GT = os.path.join(BASE, 'gt1030')
F = r'F:\weibo_cos'
OUTDIR = os.path.join(F, 'ar_run_igpu40k_b')
TRAINLOG = os.path.join(F, 'ar_igpu_runb.log')
DRIVER_LOG = os.path.join(F, 'ar_igpu_runb_driver.log')
NOTE = r'D:\work\collab\shared\IGPU_RUNB_RESULT.md'
CKPT_START = os.path.join(F, 'ar_run_igpu40k', 'ckpt_30000.pt')
STEPS = 20000


def log(msg):
    line = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with io.open(DRIVER_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    log('=== run B start（从 ckpt_30000 续，20000 步，lr 2e-5→3e-6）===')
    cmd = [PY, os.path.join(GT, 'ar_v2_train.py'),
           '--ckpt', CKPT_START, '--steps', str(STEPS),
           '--lr', '2e-5', '--lr-min', '3e-6', '--warmup', '200',
           '--save-every', '2500', '--log', TRAINLOG,
           '--out-dir', OUTDIR, '--val-frac', '0.02', '--shuffle', '1']
    log('RUN ' + ' '.join(cmd))
    t0 = time.time()
    with io.open(DRIVER_LOG, 'a', encoding='utf-8') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    log('训练结束 rc=%d 用时 %.1f 小时' % (rc, (time.time() - t0) / 3600))

    last = os.path.join(OUTDIR, 'ckpt_last.pt')
    if not os.path.exists(last):
        cands = sorted(x for x in os.listdir(OUTDIR) if x.startswith('ckpt_') and x.endswith('.pt'))
        last = os.path.join(OUTDIR, cands[-1]) if cands else CKPT_START
    png = os.path.join(F, 'ar_sample_runb.png')
    subprocess.call([PY, os.path.join(BASE, 'sample_ar.py'), '--ar', last,
                     '--vqvae', r'D:\work\cloud_salvage\ScPeP7\vqvae_V32.pt',
                     '--out', png, '--n', '8', '--top-k', '100', '--seed', '0'])
    tail, metrics = '', ''
    try:
        with io.open(TRAINLOG, encoding='utf-8', errors='replace') as f:
            tail = '\n'.join(f.read().splitlines()[-10:])
    except Exception as e:
        tail = str(e)
    try:
        r = subprocess.run([PY, os.path.join(GT, 'ar_token_analysis.py'),
                            '--tokens', png[:-4] + '.tokens.pt', '--label', 'runB'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        metrics = (r.stdout or '').strip()
    except Exception as e:
        metrics = repr(e)
    with io.open(NOTE, 'w', encoding='utf-8') as f:
        f.write('# 核显 run B（从 ckpt_30000 续 20000 步）\n\n'
                '- 起点 `ckpt_30000.pt`（val **5.1963**，基线 ckpt_20000 是 5.3551）\n'
                '- 配方 lr 2e-5 → 3e-6（warmup 200 + cosine），NaN 守卫（非有限 loss/grad 跳过）\n'
                '- 产物 `%s\\ckpt_32500 … 50000.pt`、采样 `%s`\n\n'
                '## 训练日志尾\n```\n%s\n```\n\n## token 指标\n```\n%s\n```\n'
                % (OUTDIR, png, tail, metrics))
    log('done，笔记写好: ' + NOTE)


if __name__ == '__main__':
    main()
