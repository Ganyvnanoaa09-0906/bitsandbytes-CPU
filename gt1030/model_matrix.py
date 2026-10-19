"""Run the universal iGPU trainer across a matrix of tiny models of different architectures.

For each local model directory under D:\\deepsleep\\tiny_models: run a full fine-tune smoke (6 steps) and
a LoRA smoke (6 steps), parse the RESULT lines, and print a markdown table + JSON. This is the
generality check: nothing here is architecture-specific.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ROOT = r'D:\deepsleep\tiny_models'
TRAINER = r'D:\work\bitsandbytes-CPU\gt1030\igpu_train.py'
PY = sys.executable
RESULT = re.compile(
    r'RESULT (.+?) \| (\w+) \| (\S+) \| steps=(\d+) \| ([\d.]+) s/step \| loss ([\d.]+) -> ([\d.]+) \| finite=(\w+)')


def run_one(model_dir, mode, steps=6):
    cmd = [PY, TRAINER, '--model', model_dir, '--mode', mode, '--steps', str(steps),
           '--tokens', '64', '--log-every', '100']
    t0 = __import__('time').time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    dt = __import__('time').time() - t0
    out = proc.stdout + '\n' + proc.stderr
    arch = '?'
    for line in out.splitlines():
        m = RESULT.search(line)
        if m:
            return dict(model=m.group(1), mode=m.group(2), arch=m.group(3), steps=int(m.group(4)),
                        step_s=float(m.group(5)), loss0=float(m.group(6)), loss1=float(m.group(7)),
                        finite=m.group(8) == 'YES', wall=dt, error=None)
    err = ' | '.join(l.strip() for l in out.splitlines()[-4:] if l.strip())[:300]
    return dict(model=model_dir, mode=mode, arch=arch, steps=0, step_s=0.0, loss0=0.0,
                loss1=0.0, finite=False, wall=dt, error=err)


def main() -> int:
    models = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    rows = []
    for m in models:
        for mode in ('full', 'lora'):
            r = run_one(os.path.join(ROOT, m), mode)
            rows.append(r)
            if r['error']:
                print('%-52s %-5s FAIL %s' % (m[:52], mode, r['error'][:160]), flush=True)
            else:
                print('%-52s %-5s ok  %.3f s/step  loss %.3f -> %.3f  finite=%s'
                      % (m[:52], mode, r['step_s'], r['loss0'], r['loss1'], r['finite']), flush=True)
    out_json = r'D:\work\bitsandbytes-CPU\i5build\model_tests\igpu_generality_matrix.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(rows, open(out_json, 'w', encoding='utf-8'), indent=1)
    ok = sum(1 for r in rows if r['finite'])
    print()
    print('matrix: %d runs, %d finite-ok, %d failed' % (len(rows), ok, len(rows) - ok))
    print('saved', out_json)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
