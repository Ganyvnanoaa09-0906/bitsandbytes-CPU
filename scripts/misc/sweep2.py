# -*- coding: utf-8 -*-
"""sweep2.py — 把采样参数推到边界。

第一轮结论（与直觉相反）：
  k 30 → 100 → 300 单调改善（−13.96 → −11.55 → −10.53），码本 55.9% → 79.2% → 93.8%
  降温度是灾难（t0.6 码本塌到 5.9%，高频 −32.66）
  ⇒ **模型的分布铺得太开、不够自信**；截断/锐化把它逼到错误模态上。
  ⇒ 那就继续放开：k=512 / 1024（不截断），并试温度略 >1。

本轮网格：k ∈ {512, 1024} × temp ∈ {1.0, 1.15}，外加 k=1024/temp=1.3 探上界。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, subprocess, sys

BB = r'D:\work\bitsandbytes-CPU'
PY = sys.executable
CK = r'D:\work\cloud_salvage\ScPeP7\ckpt_20000.pt'
VQ = r'D:\work\cloud_salvage\ScPeP7\vqvae_V32.pt'
OUT = r'D:\work\cloud_results\SWEEP2'
os.makedirs(OUT, exist_ok=True)

GRID = [(1.0, 512), (1.0, 1024), (1.15, 1024), (1.15, 512), (1.30, 1024)]

print('%-16s %-4s %9s %9s %9s   %s' % ('config', 'rc', 'low', 'mid', 'HIGH', 'codebook'))
rows = []
for temp, k in GRID:
    tag = 't%.2f_k%04d' % (temp, k)
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
    print('%-16s %-4s %9s %9s %9s   %s' % (tag, rc, vals[0], vals[1], vals[2], cb), flush=True)
    rows.append((tag, vals[2], cb))

print()
print('参照：天花板 −7.66 | TF@20000 −8.73 | k100 −11.55 | k300 −10.53')
best = None
for tag, hf, cb in rows:
    try:
        v = float(hf)
        if best is None or v > best[1]:
            best = (tag, v)
    except Exception:
        pass
if best:
    print('本轮最好：%s (%s dB)' % best)
print('产物：%s' % OUT)
