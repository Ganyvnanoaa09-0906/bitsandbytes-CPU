# -*- coding: utf-8 -*-
"""run_clean_new.py — 清洗三个未进过管线的原始目录（"数据是杠杆"判决后的执行）

判决依据（见 HANDOFF_2026-09-19.md）：`ar_halfdata` 同配置同 seed 只用一半数据，
从 step 3600 起稳定落后 0.07~0.18 nats ⇒ **数据是杠杆**。

数据盘点（data_overlap2.py，8/12/16 位哈希前缀三种粒度一致）：
    nsfw_dataset   28,000 张   与已清洗集重叠 0%
    civitai_nsfw    6,537 张   重叠 0%
    real_train_300  5,600 张   重叠 0%
                          ----------
                          40,137 张

参数与原管线一致（从 clean_* 的 clean_report.json 读出）：
    res=512  batch=8  detect=ssdlite   ← 且【没有】--require-person
    （丢弃原因里没有 no_person，印证了这点）

500 张试跑：保留率 98.4%、117 ms/张 ⇒ 预计 1.3 小时拿到 ~39,500 张新图。
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
import subprocess, sys, os, time, json

ROOT = r'D:\work\bitsandbytes-CPU'
PY = sys.executable
AP = os.path.join(ROOT, '.autopilot')

JOBS = [
    ('nsfw_dataset',   r'D:\work\sujvji\nsfw_dataset',   'clean_new_nsfw'),
    ('civitai_nsfw',   r'D:\work\sujvji\civitai_nsfw',   'clean_new_civitai'),
    ('real_train_300', r'D:\work\sujvji\real_train_300', 'clean_new_realtrain'),
]

env = dict(os.environ, PYTHONIOENCODING='utf-8')
print('=' * 88)
print('清洗未用原始图（数据是杠杆 ⇒ 执行分支）')
print('=' * 88, flush=True)

t_all = time.time()
summary = []
for name, src, outname in JOBS:
    out = os.path.join(AP, outname)
    n_src = sum(1 for dp, _d, fs in os.walk(src)
                for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')))
    print()
    print('### %-16s 源 %6d 张 -> %s' % (name, n_src, outname), flush=True)
    if os.path.isdir(out) and len(os.listdir(os.path.join(out, 'images'))) > 100:
        print('  （已有结果，跳过）', flush=True)
        continue
    t0 = time.time()
    r = subprocess.run([PY, '-u', os.path.join(ROOT, 'fast_clean.py'),
                        '--data', src, '--out', out,
                        '--res', '512', '--batch', '8', '--threads', '6',
                        '--detect', 'ssdlite', '--save'],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    dt = time.time() - t0
    tail = [l for l in (r.stdout or '').strip().split('\n') if l.strip()][-3:]
    print('\n'.join('  ' + l for l in tail), flush=True)
    if r.returncode != 0:
        print('  !! 退出码 %d\n%s' % (r.returncode, (r.stderr or '')[-800:]), flush=True)
    # 读结果报告
    rp = os.path.join(out, 'clean_report.json')
    if os.path.exists(rp):
        rep = json.load(open(rp, encoding='utf-8'))
        print('  => 保留 %d / 拒绝 %d   %.1f ms/图   耗时 %.0f min'
              % (rep['kept'], rep['rejected'], rep['ms_per_img'], dt / 60), flush=True)
        summary.append((name, rep['images'], rep['kept'], rep['rejected']))
    else:
        print('  !! 没找到 clean_report.json', flush=True)

print()
print('=' * 88)
print('汇总')
print('=' * 88)
tot_kept = 0
for name, n, k, rej in summary:
    print('  %-18s 输入 %6d  保留 %6d  拒绝 %5d  (%.1f%%)'
          % (name, n, k, rej, 100 * k / max(n, 1)))
    tot_kept += k
print('  ' + '-' * 70)
print('  新增可训练图 %d 张   原有 29113 张   =>  合计 %d 张  (+%.0f%%)'
      % (tot_kept, 29113 + tot_kept, 100 * tot_kept / 29113))
print('  总耗时 %.1f min' % ((time.time() - t_all) / 60))
print()
print('下一步：')
print('  ① 与已有 29113 张去重（dHash，dedup_images.py）')
print('  ② 合并 paths 清单 -> tokenize_dataset.py 产出新 tokens')
print('  ③ 同配置同 seed 重训，与 ar_flat256 比匹配步数的 val')
print('DONE')
