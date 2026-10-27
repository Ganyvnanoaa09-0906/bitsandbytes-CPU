# -*- coding: utf-8 -*-
"""看着 40k 通宵跑：每到一个 5k 的 ckpt 就采样 4 张 + 算 token 指标，追加进进度笔记。

    python gt1030/_igpu_40k_watch.py        # 独立进程跑，和训练互不干扰（每次采样约占 27s GPU）

产出：
  F:\\weibo_cos\\igpu40k_samples\\s<step>.png / .tokens.pt
  D:\\work\\collab\\shared\\IGPU_40K_PROGRESS.md   （逐行追加指标）
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
OUTDIR = os.path.join(F, 'ar_run_igpu40k')
SMPDIR = os.path.join(F, 'igpu40k_samples')
NOTE = r'D:\work\collab\shared\IGPU_40K_PROGRESS.md'
VQVAE = r'D:\work\cloud_salvage\ScPeP7\vqvae_V32.pt'
CKPTS = [25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]
TRAINLOG = os.path.join(F, 'ar_igpu_40k.log')


def log(msg):
    line = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with io.open(NOTE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def train_alive():
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              "Get-Process python -ErrorAction SilentlyContinue | "
                              "Where-Object { $_.Id -in 22468 } | "
                              "Select-Object -ExpandProperty Id"],
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
        return '22468' in (out.stdout or '')
    except Exception:
        return True


def main():
    os.makedirs(SMPDIR, exist_ok=True)
    if not os.path.exists(NOTE):
        with io.open(NOTE, 'w', encoding='utf-8') as f:
            f.write('# 核显 40k 训练进度（每 5k 步自动采样 + token 指标）\n\n'
                    '对照：真实数据 边际熵 9.55 / 每图码 417 / KL 0 / 邻接 h 9.9% v 9.2%\n'
                    '起点 20k：边际熵 6.77 / 每图码 108 / KL 1.84 / 邻接 h 21.4% v 6.0%\n\n'
                    '```\n')
    log('watcher start')
    for step in CKPTS:
        ck = os.path.join(OUTDIR, 'ckpt_%d.pt' % step)
        t0 = time.time()
        while not os.path.exists(ck):
            if not train_alive():
                log('训练进程已退出，未见 ckpt_%d，watcher 结束' % step)
                return
            if time.time() - t0 > 6 * 3600:
                log('等 ckpt_%d 超时 6h，结束' % step)
                return
            time.sleep(60)
        png = os.path.join(SMPDIR, 's%d.png' % step)
        try:
            subprocess.run([PY, os.path.join(BASE, 'sample_ar.py'), '--ar', ck,
                            '--vqvae', VQVAE, '--out', png, '--n', '4',
                            '--top-k', '100', '--seed', '0'],
                           check=True, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
            tok = png[:-4] + '.tokens.pt'
            r = subprocess.run([PY, os.path.join(GT, 'ar_token_analysis.py'),
                                '--tokens', tok, '--label', '%d' % step],
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
            line = (r.stdout or '').strip().splitlines()
            log(line[-1] if line else '  %d 分析无输出' % step)
        except Exception as e:
            log('  %d 采样/分析失败: %r' % (step, e))
    log('全部完成')
    try:
        with io.open(TRAINLOG, encoding='utf-8', errors='replace') as f:
            tail = '\n'.join(f.read().splitlines()[-6:])
        log('训练日志尾:\n' + tail)
    except Exception:
        pass


if __name__ == '__main__':
    main()
