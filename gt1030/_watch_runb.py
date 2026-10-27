# -*- coding: utf-8 -*-
"""run B 的看护：每次出现新的 ckpt 就采样 4 张 + 算 token 指标，追加进笔记。

    python gt1030/_watch_runb.py

（编码 bug 已修：子进程输出强制 utf-8，之前 GBK 解码失败导致"分析无输出"）
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
OUTDIR = os.environ.get('ARW_OUTDIR', os.path.join(F, 'ar_run_igpu40k_b'))
SMPDIR = os.environ.get('ARW_SMPDIR', os.path.join(F, 'runb_samples'))
NOTE = os.environ.get('ARW_NOTE', r'D:\work\collab\shared\IGPU_RUNB_PROGRESS.md')
VQVAE = r'D:\work\cloud_salvage\ScPeP7\vqvae_V32.pt'
TAG = os.environ.get('ARW_TAG', 'runB')
_ck = os.environ.get('ARW_CKPTS', '')
CKPTS = [int(x) for x in _ck.split(',')] if _ck else \
    [32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000]
TRAINLOG = os.environ.get('ARW_LOG', os.path.join(F, 'ar_igpu_runb.log'))


def log(msg):
    line = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with io.open(NOTE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def trainer_alive():
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                              "Select-Object -ExpandProperty CommandLine"],
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
        return 'ar_v2_train.py' in (out.stdout or '')
    except Exception:
        return True


def main():
    os.makedirs(SMPDIR, exist_ok=True)
    if not os.path.exists(NOTE):
        with io.open(NOTE, 'w', encoding='utf-8') as f:
            f.write('# ' + TAG + ' 进度（每 2.5k 步采样 + token 指标）\n\n'
                    '真实数据: 边际熵 9.55 / 每图码 417 / 邻接一致 h 9.9% v 9.2%\n'
                    '20k 基线: 6.77 / 108 / 21.4% 6.0%   |   30k(run A): 7.28 / 140 / 15.5% 7.3%\n\n'
                    '```\n')
    log('watcher start (%s)' % TAG)
    for step in CKPTS:
        ck = os.path.join(OUTDIR, 'ckpt_%d.pt' % step)
        t0 = time.time()
        while not os.path.exists(ck):
            if not trainer_alive():
                log('训练进程已退出，未见 ckpt_%d，结束' % step); return
            if time.time() - t0 > 8 * 3600:
                log('等 ckpt_%d 超时，结束' % step); return
            time.sleep(45)
        png = os.path.join(SMPDIR, 'b%d.png' % step)
        try:
            subprocess.run([PY, os.path.join(BASE, 'sample_ar.py'), '--ar', ck,
                            '--vqvae', VQVAE, '--out', png, '--n', '4',
                            '--top-k', '100', '--seed', '0'],
                           check=True, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
            r = subprocess.run([PY, os.path.join(GT, 'ar_token_analysis.py'),
                                '--tokens', png[:-4] + '.tokens.pt', '--label', '%d' % step],
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
            ln = (r.stdout or '').strip().splitlines()
            log(ln[-1] if ln else '%d 分析无输出' % step)
        except Exception as e:
            log('%d 采样/分析失败: %r' % (step, e))
    log('全部完成')
    try:
        with io.open(TRAINLOG, encoding='utf-8', errors='replace') as f:
            log('日志尾:\n' + '\n'.join(f.read().splitlines()[-5:]))
    except Exception:
        pass


if __name__ == '__main__':
    main()
