# -*- coding: utf-8 -*-
"""sweep_sampling.py — 扫采样参数（温度 × top-k）。

动机：我们从没扫过采样参数 —— `top-k=100, temperature=1.0` 是当年随手定的。
      而 FREE 的失败正是"采样噪声累积导致漂移"：
        · 温度太低 / 截断太狠 ⇒ 塌成均值（贪心实测码本只剩 4/1024）
        · 温度太高 / 截断太松 ⇒ 噪声累积、漂离数据流形（现在这一步）
      ⇒ 中间大概率有甜点，而且**不改模型、不训练、纯免费**。

判据：高频缺口 + **码本使用率**（塌缩会掉）+ **看图**。
"""
import os, subprocess, sys

BB = r'D:\work\bitsandbytes-CPU'
PY = sys.executable
CK = r'D:\work\cloud_salvage\ScPeP7\ckpt_20000.pt'
VQ = r'D:\work\cloud_salvage\ScPeP7\vqvae_V32.pt'
OUT = r'D:\work\cloud_results\SWEEP'
os.makedirs(OUT, exist_ok=True)

GRID = [(1.0, 30), (1.0, 100), (1.0, 300),
        (0.8, 100), (0.8, 300),
        (0.6, 300)]

print('%-16s %-8s %10s %10s %10s   %s' % ('config', 'rc', 'low', 'mid', 'HIGH', 'codebook/entropy'))
rows = []
for temp, k in GRID:
    tag = 't%.1f_k%03d' % (temp, k)
    png = os.path.join(OUT, tag + '.png')
    log = os.path.join(OUT, tag + '.log')
    with open(log, 'w', encoding='utf-8') as f:
        rc = subprocess.run([PY, '-u', 'sample_ar.py', '--ar', CK, '--vqvae', VQ,
                             '--out', png, '--n', '16', '--top-k', str(k),
                             '--temperature', str(temp), '--threads', '6'],
                            cwd=BB, stdout=f, stderr=subprocess.STDOUT).returncode
    txt = open(log, encoding='utf-8', errors='replace').read()
    cb = ''
    for l in txt.splitlines():
        if '个码' in l:
            cb = l.split('生成 token:')[-1].strip()
    glog = os.path.join(OUT, tag + '.gap')
    with open(glog, 'w', encoding='utf-8') as f:
        subprocess.run([PY, '-u', 'spectrum_gap.py', '--vqvae', VQ, '--sample', png,
                        '--n', '16', '--threads', '6'], cwd=BB, stdout=f,
                       stderr=subprocess.STDOUT)
    vals = ['-', '-', '-']
    for l in open(glog, encoding='utf-8', errors='replace'):
        if l.startswith('AR 生成'):
            p = l.split()
            if len(p) >= 5:
                vals = p[2:5]
    print('%-16s %-8s %10s %10s %10s   %s' % (tag, rc, vals[0], vals[1], vals[2], cb), flush=True)
    rows.append((tag, vals[2], cb))

print()
print('参考：天花板高频 −7.66 dB ；TF@20000 −8.73 dB ；未扫时(t1.0_k100) −11.55 dB')
best = None
for tag, hf, cb in rows:
    try:
        v = float(hf)
        if best is None or v > best[1]:
            best = (tag, v)
    except Exception:
        pass
print('本网格最好：%s (%s dB)' % best if best else '无')
print('产物目录：%s' % OUT)
