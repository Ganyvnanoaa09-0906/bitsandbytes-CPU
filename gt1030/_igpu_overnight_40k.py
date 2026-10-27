# -*- coding: utf-8 -*-
"""核显 4 万步通宵训练的驱动器（独立进程，脱离 agent 会话）。

    python gt1030/_igpu_overnight_40k.py

做的事：
  1) 从 ckpt_20000 续训 40000 步（lr 3e-5 → 5e-6，warmup 300，每 5000 步存 ckpt）
  2) 结束后用 sample_ar 采 4 张图
  3) 把日志尾巴 + 采样统计写到 D:\\work\\collab\\shared\\IGPU_40K_RESULT.md
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
LOGDIR = r'F:\weibo_cos'
TRAINLOG = os.path.join(LOGDIR, 'ar_igpu_40k.log')
DRIVER_LOG = os.path.join(LOGDIR, 'ar_igpu_40k_driver.log')
NOTE = r'D:\work\collab\shared\IGPU_40K_RESULT.md'
CKPT_START = r'F:\weibo_cos\ar_run_full3\ckpt_20000.pt'
OUTDIR = r'F:\weibo_cos\ar_run_igpu40k'
STEPS = 40000


def log(msg):
    line = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with io.open(DRIVER_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd, tag):
    log('RUN %s' % ' '.join(cmd))
    t0 = time.time()
    with io.open(DRIVER_LOG, 'a', encoding='utf-8') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    log('%s done rc=%d in %.0f min' % (tag, rc, (time.time() - t0) / 60))
    return rc


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    env_note = 'AR_ARENA_MB=%s' % os.environ.get('AR_ARENA_MB', 'unset')
    log('=== iGPU 40k overnight start (%s) ===' % env_note)
    cmd = [PY, os.path.join(GT, 'ar_v2_train.py'),
           '--ckpt', CKPT_START, '--steps', str(STEPS),
           '--lr', '3e-5', '--lr-min', '5e-6', '--warmup', '300',
           '--save-every', '5000', '--log', TRAINLOG,
           '--out-dir', OUTDIR, '--val-frac', '0.02', '--shuffle', '1']
    rc = run(cmd, 'train40k')

    last = os.path.join(OUTDIR, 'ckpt_last.pt')
    if not os.path.exists(last):
        cands = [f for f in os.listdir(OUTDIR) if f.startswith('ckpt_') and f.endswith('.pt')]
        cands.sort()
        last = os.path.join(OUTDIR, cands[-1]) if cands else CKPT_START
    png = os.path.join(LOGDIR, 'ar_sample_igpu40k.png')
    run([PY, os.path.join(BASE, 'sample_ar.py'), '--ar', last,
         '--vqvae', r'D:\work\cloud_salvage\ScPeP7\vqvae_V32.pt',
         '--out', png, '--n', '4', '--top-k', '100', '--seed', '0'], 'sample')

    tail = ''
    try:
        with io.open(TRAINLOG, encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
        tail = '\n'.join(lines[-12:])
    except Exception as e:
        tail = str(e)
    text = ('# 核显 4 万步续训（2026-10-25 晚 → 26 日早）\n\n'
            '- 起点：`ckpt_20000.pt`（CPU 20k，val 5.3560）\n'
            '- 配方：lr 3e-5 → 5e-6（warmup 300 + cosine）、bat 2×1024 token、尾 2% 作 val\n'
            '- 预期：40000 步 ≈ 82M token，累计 ~123M token（~6.4 epoch），13.7 小时\n'
            '- 速度：1.216 s/step = 1684 tok/s（CPU 同模型 569 tok/s ⇒ 2.96×）\n\n'
            '## 训练日志尾部\n```\n%s\n```\n\n'
            '## 采样\n- `%s`\n\n'
            '## 备注\n- 若采样仍是色块：属正常（Chinchilla 口径需要 ~610M token，这次只到 6.4 epoch）\n'
            '- 中间 ckpt：`%s\\ckpt_25000/30000/.../60000.pt`\n' % (tail, png, OUTDIR))
    with io.open(NOTE, 'w', encoding='utf-8') as f:
        f.write(text)
    log('=== done, note written ===')


if __name__ == '__main__':
    main()
