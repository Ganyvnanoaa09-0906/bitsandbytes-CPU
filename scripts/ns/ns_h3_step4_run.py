"""H3 round 2, step 4: parameterise the NS iteration count in the Muon A/B harness and run
the NS=3 arm against the existing NS=5 record.

Discovery while setting this up, worth recording: train_ar_v2.py has no Muon path at all.
Muon exists only in ab_muon_v2.py, the standalone A/B harness. So the trainer cannot simply
be given a flag; the harness has to carry the variable.

That also means the existing 2000-step Muon arm is an NS=5 record on exactly the protocol
this arm will use, so only one new arm is needed. The comparison is:

    muon      NS=5, already recorded (2000 steps; the 600-step portion is the comparable slice)
    muon3     NS=3, to be run now, same harness, same 600 steps

Protocol held fixed: same tokens, same model, same seed, same learning rate, same batch,
same sequence length, same validation set and frequency. One variable.

Screening length is 600 steps, as the plan permits for a first pass, and the arms are judged
on validation loss and optimiser time, not on training loss, whose per-step standard deviation
is about 0.58.

Note on the ceiling: the microbenchmark says NS 5->3 saves 763 ms of a 3952 ms step, i.e. a
1.24x ceiling on the full step. That is a ceiling, not a prediction; the plan explicitly warns
against quoting the local NS ratio as an end-to-end speedup, and this run is what would
measure the real one. Nothing here presumes 1.24x will be realised.
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import io
import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

ROOT = r'D:\work\bitsandbytes-CPU'
os.chdir(ROOT)
P = os.path.join(ROOT, 'ab_muon_v2.py')
s = io.open(P, encoding='utf-8').read()

if not os.path.exists(P):
    print('  ? ab_muon_v2.py 不存在')
    sys.exit(1)

# show the current knobs
print('  当前的 NS 配置:')
for kw in ('NS_STEPS', 'STEPS', 'LR_MUON'):
    for m in __import__('re').finditer(r'^%s\s*=.*$' % kw, s, __import__('re').M):
        print('    %s' % m.group(0).strip())

# 1. make NS_STEPS overridable from argv, add a 'muon3' arm
if 'muon3' not in s:
    if 'NS_STEPS = 5' in s:
        s = s.replace('NS_STEPS = 5',
                      'NS_STEPS = int(os.environ.get("NS_ITERS", "5"))', 1)
        print('  + NS_STEPS 改为读环境变量 NS_ITERS（默认 5 ⇒ 基线不变）')
    # find the arm loop and add a third arm
    import re
    m = re.search(r'for\s+(\w+)\s+in\s*\(([^)]*adamw[^)]*muon[^)]*)\)', s)
    if m:
        s = s[:m.end(2)] + ", 'muon3'" + s[m.end(2):]
        print('  + 已加入 muon3 臂')
        # make make_opt understand muon3
        s = s.replace("if name == 'muon':", "if name in ('muon', 'muon3'):", 1)
        print('  + make_opt 已识别 muon3')
    else:
        print('  ? 臂列表没找到，需要手工加')
    io.open(P, 'w', encoding='utf-8', newline='').write(s)
    import py_compile
    try:
        py_compile.compile(P, doraise=True)
        print('  ✓ 语法通过')
    except Exception as e:
        print('  ✗ %s' % e)
        sys.exit(1)
else:
    print('  已改过')

print()
print('  现在是 [t] 之前的状态:')
print('    muon   臂 = NS=5（2000 步已记录，取其前 600 步作对照）')
print('    muon3  臂 = NS=3（本次新跑 600 步）')
print()
print('  启动 muon3 臂（NS_ITERS=3），约 1 小时。原始日志: ns_ab_muon3.log')
env = dict(os.environ)
env['NS_ITERS'] = '3'
env['PYTHONIOENCODING'] = 'utf-8'
log = open(os.path.join(ROOT, 'ns_ab_muon3.log'), 'w', encoding='utf-8')
t0 = time.perf_counter()
p = subprocess.Popen(['python', '-u', 'ab_muon_v2.py'], stdout=log, stderr=subprocess.STDOUT,
                     env=env, cwd=ROOT)
print('  PID %d' % p.pid)
print('  开始时间 %s' % time.strftime('%H:%M:%S'))
