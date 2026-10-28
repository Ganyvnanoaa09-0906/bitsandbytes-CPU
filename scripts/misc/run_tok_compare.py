# -*- coding: utf-8 -*-
"""run_tok_compare.py — 三个分词器在同一批图（同一 N）上的可学习性对比

判决指标只用 H(x | 左邻)：本项目唯一被证明可信的条件熵（稠密表）。
三者用同一个 N、同一份 paths.json ⇒ 差异只来自分词器。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import subprocess, sys, os, time

ROOT = r'D:\work\bitsandbytes-CPU'
PY = sys.executable
N = 6000

JOBS = [
    ('C_mse (纯MSE)',         r'D:\work\cloud_results\ckpt_C_mse.pt',
     r'D:\work\bitsandbytes-CPU\.autopilot\tokC6k.pt'),
    ('P_attn (瓶颈注意力)',   r'D:\work\cloud_results\P_16x16_attn_small_17820.pt',
     r'D:\work\bitsandbytes-CPU\.autopilot\tokP6k.pt'),
    ('F_perc (感知损失)',     r'D:\work\cloud_results\F_16x16_perc_last.pt',
     r'D:\work\bitsandbytes-CPU\.autopilot\tokF6k.pt'),
]

env = dict(os.environ, PYTHONIOENCODING='utf-8')
results = []
for name, ckpt, out in JOBS:
    print('\n' + '#' * 92)
    print('# %s' % name)
    print('#' * 92, flush=True)
    t0 = time.time()
    r = subprocess.run([PY, '-u', os.path.join(ROOT, 'tok_compare.py'),
                        '--ckpt', ckpt, '--out', out, '--name', name,
                        '--limit', str(N), '--threads', '6'],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    print(r.stdout)
    if r.returncode != 0:
        print('!! 退出码 %d\n%s' % (r.returncode, (r.stderr or '')[-1500:]))
    print('耗时 %.0fs' % (time.time() - t0), flush=True)

print('\n' + '=' * 92)
print('三分词器汇总（N=%d，同一批图）' % N)
print('=' * 92)
