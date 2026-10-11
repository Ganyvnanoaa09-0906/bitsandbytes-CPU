# -*- coding: utf-8 -*-
"""eval_variants.py — 用本地 ckpt 回答"昨晚那四个采样器修改到底有没有用"。

背景：云算力烧了一整夜跑 4 个 T=1024 变体，但预算在跑到一半时耗尽，
      它们从未被评估过。ckpt 已抢救回本地 ⇒ **现在可以免费评估**。

⚠️ 必须说清的混淆：各变体存的步数不同（基线 20000 / z 18000 / td 18000 / ss 10000 / c2f 14000）
   ⇒ 不是严格的同步数对照。但**若某变体在更少步数下已经更好，那结论更强**。

判据（本项目已确立）：**自由采样的高频缺口 + 看图有没有主体**。
  val 不算数（已证明 val 好 ≠ 画质好）。
"""
import os, subprocess, sys, json, glob

BB = r'D:\work\bitsandbytes-CPU'
PY = sys.executable
VQ = r'D:\work\cloud_salvage\ScPeP7\vqvae_V32.pt'
SAL = r'D:\work\cloud_salvage'
OUT = r'D:\work\cloud_results\VARIANTS'
COARSE = r'D:\work\cloud_results\coarse16_29113.pt'
os.makedirs(OUT, exist_ok=True)

VARIANTS = [
    # name, ckpt, step, 额外参数
    ('baseline',  os.path.join(SAL, 'ScPeP7', 'ckpt_20000.pt'), 20000, []),
    ('zorder',    os.path.join(SAL, 'CrswCc', 'ckpt_18000.pt'), 18000, []),
    ('tokdrop',   os.path.join(SAL, 'XZUwKb', 'ckpt_18000.pt'), 18000, []),
    ('schedsamp', os.path.join(SAL, 'oWAZY8', 'ckpt_10000.pt'), 10000, []),
    ('coarse2f',  os.path.join(SAL, 'GafcwJ', 'ckpt_14000.pt'), 14000,
     ['--cond-pt', COARSE, '--text-dim', '64']),
]

def run(cmd, log):
    with open(log, 'w', encoding='utf-8') as f:
        return subprocess.run(cmd, cwd=BB, stdout=f, stderr=subprocess.STDOUT).returncode

results = []
for name, ck, step, extra in VARIANTS:
    if not os.path.exists(ck):
        print('  %-10s ✗ ckpt 缺失 %s' % (name, ck)); continue
    png = os.path.join(OUT, '%s_s%d.png' % (name, step))
    log = os.path.join(OUT, '%s_s%d.log' % (name, step))
    cmd = [PY, '-u', 'sample_ar.py', '--ar', ck, '--vqvae', VQ,
           '--out', png, '--n', '16', '--top-k', '100', '--temperature', '1.0',
           '--threads', '6'] + extra
    rc = run(cmd, log)
    txt = open(log, encoding='utf-8', errors='replace').read()
    stat = [l for l in txt.splitlines() if '码' in l or '熵' in l]
    print('  %-10s s%-6d rc=%s  %s' % (name, step, rc, stat[0].strip() if stat else '(无统计)'), flush=True)
    results.append((name, step, png, stat[0].strip() if stat else ''))

print()
print('=== 频域对比（同一分词器、同一解码器）===')
print('  %-10s %-7s %10s %10s %10s' % ('variant', 'step', 'low', 'mid', 'HIGH'))
for name, step, png, _ in results:
    if not os.path.exists(png):
        continue
    log = png.replace('.png', '.gap.log')
    with open(log, 'w', encoding='utf-8') as f:
        subprocess.run([PY, '-u', 'spectrum_gap.py', '--vqvae', VQ, '--sample', png,
                        '--n', '16', '--threads', '6'], cwd=BB, stdout=f,
                       stderr=subprocess.STDOUT)
    line = ''
    for l in open(log, encoding='utf-8', errors='replace'):
        if l.startswith('AR 生成'):
            line = l
    parts = line.split()
    if len(parts) >= 5:
        print('  %-10s %-7d %10s %10s %10s' % (name, step, parts[2], parts[3], parts[4]))
    else:
        print('  %-10s %-7d 解析失败: %s' % (name, step, line.strip()[:60]))

print()
print('参考：天花板（真值 token 解码）高频 = −7.66 dB')
print('产物目录：%s' % OUT)
