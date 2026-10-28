# -*- coding: utf-8 -*-
"""wait_then_pipeline.py — 等清洗进程结束，然后自动跑新数据管线

清洗是长时间任务（~1 小时），管线（去重 → token 化新图 → 合并）又要 1 小时。
串起来自动跑，不必人工轮询。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, subprocess, sys, time

ROOT = r'D:\work\bitsandbytes-CPU'
AP = os.path.join(ROOT, '.autopilot')
PY = sys.executable
TARGET = 40137
DIRS = [os.path.join(AP, d) for d in
        ('clean_new_nsfw', 'clean_new_civitai', 'clean_new_realtrain')]


def processed():
    n = 0
    for d in DIRS:
        if not os.path.isdir(d):
            continue
        for dp, _dn, fns in os.walk(os.path.join(d, 'images')):
            n += sum(1 for f in fns if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        rj = os.path.join(d, 'rejected.jsonl')
        if os.path.exists(rj):
            with open(rj, encoding='utf-8', errors='replace') as f:
                n += sum(1 for _ in f)
    return n


print('等清洗结束（目标 %d 张）...' % TARGET, flush=True)
t0 = time.time()
last = -1
while True:
    n = processed()
    if n != last:
        print('  %s  %d/%d  %.1f%%' % (time.strftime('%H:%M:%S'), n, TARGET,
                                       100 * n / TARGET), flush=True)
        last = n
    # 清洗进程还在吗？（找 fast_clean 的 cmdline）
    alive = False
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                              "Select-Object -ExpandProperty CommandLine"],
                             capture_output=True, text=True, timeout=60).stdout or ''
        alive = 'fast_clean' in out
    except Exception:
        alive = True
    if not alive and n >= TARGET * 0.98:
        break
    if time.time() - t0 > 6 * 3600:
        print('  ⚠️ 等待超时（6 小时），强行进入管线', flush=True)
        break
    time.sleep(60)

print('\n清洗结束：%d 张，用时 %.1f min' % (processed(), (time.time() - t0) / 60), flush=True)
print('=' * 88, flush=True)
r = subprocess.run([PY, '-u', os.path.join(ROOT, 'pipeline_newdata.py')],
                   cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                   capture_output=True, text=True, encoding='utf-8', errors='replace')
print(r.stdout, flush=True)
if r.returncode != 0:
    print('!! 管线退出码 %d\n%s' % (r.returncode, (r.stderr or '')[-2000:]), flush=True)
