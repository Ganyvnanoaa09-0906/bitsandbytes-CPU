# -*- coding: utf-8 -*-
"""eval_variants_k300.py — 在【更好的采样参数下】重测四个变体。

为什么要重测：我先前"四个变体全比基线差"的结论是在 `top-k=100` 下得到的。
而扫参发现 **k=300 让基线好 1 dB**（−11.55 → −10.53），码本 79% → 93.8%。
⇒ 采样操作点变了，**排名可能变**（例如 token dropout 那种"教模型纠错"的改造，
   可能恰好更依赖宽的采样分布）。

纪律：与 eval_variants.py 完全同口径，**只改 top-k**，便于直接对比。
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
VQ = r'D:\work\cloud_salvage\ScPeP7\vqvae_V32.pt'
SAL = r'D:\work\cloud_salvage'
OUT = r'D:\work\cloud_results\VARIANTS_K300'
COARSE = r'D:\work\cloud_results\coarse16_29113.pt'
K = int(os.environ.get('SWEEP_K', '300'))
os.makedirs(OUT, exist_ok=True)

VARIANTS = [
    ('baseline',  os.path.join(SAL, 'ScPeP7', 'ckpt_20000.pt'), 20000, []),
    ('zorder',    os.path.join(SAL, 'CrswCc', 'ckpt_18000.pt'), 18000, []),
    ('tokdrop',   os.path.join(SAL, 'XZUwKb', 'ckpt_18000.pt'), 18000, []),
    ('schedsamp', os.path.join(SAL, 'oWAZY8', 'ckpt_10000.pt'), 10000, []),
    ('coarse2f',  os.path.join(SAL, 'GafcwJ', 'ckpt_14000.pt'), 14000,
     ['--cond-pt', COARSE, '--text-dim', '64']),
]

print('top-k = %d' % K)
print('%-10s %-7s %9s %9s %9s   %s' % ('variant', 'step', 'low', 'mid', 'HIGH', 'codebook'))
for name, ck, step, extra in VARIANTS:
    if not os.path.exists(ck):
        print('  %-10s ✗ ckpt 缺失' % name); continue
    png = os.path.join(OUT, '%s_k%d.png' % (name, K))
    log = os.path.join(OUT, '%s_k%d.log' % (name, K))
    with open(log, 'w', encoding='utf-8') as f:
        rc = subprocess.run([PY, '-u', 'sample_ar.py', '--ar', ck, '--vqvae', VQ,
                             '--out', png, '--n', '16', '--top-k', str(K),
                             '--temperature', '1.0', '--threads', '6'] + extra,
                            cwd=BB, stdout=f, stderr=subprocess.STDOUT).returncode
    txt = open(log, encoding='utf-8', errors='replace').read()
    cb = ''
    for l in txt.splitlines():
        if '个码' in l:
            cb = l.split('生成 token:')[-1].strip()
    glog = os.path.join(OUT, '%s_k%d.gap' % (name, K))
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
    print('%-10s %-7d %9s %9s %9s   %s' % (name, step, vals[0], vals[1], vals[2], cb), flush=True)
    if rc != 0:
        print('   (rc=%s — 该臂采样失败，见 %s)' % (rc, log))

print()
print('对比：k=100 时 基线 −11.55 / zorder −14.97 / tokdrop −23.47 / schedsamp −16.65')
print('天花板 −7.66 ；TF@20000 −8.73')
print('产物：%s' % OUT)
