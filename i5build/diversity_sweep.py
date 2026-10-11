"""Match the sampler's diversity to the DATA's diversity -- a measurable target.

The ruler problem solved itself: no external model needed. The AR model's own
token statistics can be compared against the ground-truth token statistics:

    ground truth : codebook 50.6% used, entropy 82.7% of max, top code 5.03%
    AR samples   : codebook 23.4% used, entropy 72.1% of max, top code 4.00%

So the model's output distribution is markedly NARROWER than the data's. That is
a concrete, in-domain definition of the quality gap, and it gives a target to
optimise: raise codebook usage and entropy toward the ground-truth values by
tuning sampling only (free), before spending hours on capacity or a finer grid.

Sweep temperature x top-k on the usable conditional path (50% prefix), parse the
sampler's own statistics line, and keep the PNGs so the best can be eyeballed.
"""
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')
HERE = r'D:\work\bitsandbytes-CPU'
OUT = os.path.join(HERE, 'i5build', 'diversity')
os.makedirs(OUT, exist_ok=True)
AR = os.path.join(HERE, 'i5build', 'loop_ar3', 'ckpt.pt', 'ckpt_last.pt')
VQ = r'D:\work\cloud_salvage\ScPeP7\vqvae_16x16_cmse.pt'
TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt'

# ground-truth benchmark measured with --decode-only
GT = dict(code=50.6, ent=82.7, top=5.03)

GRID = [(1.0, 64), (1.0, 256), (1.1, 256), (1.25, 256), (1.25, 1024),
        (1.4, 1024)]
PAT = re.compile(r'使用\s*(\d+)/1024\s*个码\s*\(([\d.]+)%\).*?熵\s*([\d.]+)/[\d.]+\s*\(([\d.]+)%\)')

print('目标（真实 token 的统计）: 码本 %.1f%%  熵 %.1f%%' % (GT['code'], GT['ent']))
print('%-16s %10s %10s %10s   %s' % ('配置', '码本占比', '熵占比', '高频码', '与目标的差距'))
print('-' * 78)
rows = []
for temp, tk in GRID:
    tag = 'T%.2f_k%d' % (temp, tk)
    out = os.path.join(OUT, tag + '.png')
    cmd = [sys.executable, 'sample_ar_cond.py', '--ar', AR, '--vqvae', VQ,
           '--tokens', TOK, '--out', out, '--n', '8', '--frac', '0.5',
           '--top-k', str(tk), '--temperature', str(temp),
           '--seed', '21', '--threads', '6']
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    m = PAT.search(r.stdout or '')
    if not m:
        print('%-16s  （没解析到统计行）' % tag)
        continue
    code = float(m.group(2))
    ent = float(m.group(4))
    d = abs(code - GT['code']) + abs(ent - GT['ent'])
    rows.append((tag, code, ent, d, out))
    print('%-16s %9.1f%% %9.1f%% %10s   %.1f' % (tag, code, ent, m.group(3), d))

print('\n' + '=' * 78)
if rows:
    rows.sort(key=lambda x: x[3])
    print('最接近数据分布的配置（差距越小越好）：')
    for tag, code, ent, d, out in rows[:3]:
        print('  %-16s 码本 %.1f%%  熵 %.1f%%  差距 %.1f   -> %s'
              % (tag, code, ent, d, os.path.basename(out)))
    print('\n注意：统计接近不等于好看。图仍要看 —— 但这一轮至少有了【方向】：')
    print('      若提高温度能让码本/熵靠近数据而画面不崩 ⇒ 免费的质量提升；')
    print('      若一提温就崩 ⇒ 说明是容量/网格的限制，得训更大的模型或更细的网格。')
