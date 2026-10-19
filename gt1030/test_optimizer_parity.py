"""test_optimizer_parity.py — the iGPU training kit must offer the same optimizer experience
as the CPU training scripts.

Runs gt1030/igpu_train.py for each backend (cpu + igpu when OpenCL is present) and each
optimizer (adamw / adamw8bit / muon) on a tiny local model, checks `--check` exits 0 and the
RESULT line names the optimizer, then checks that `--save`/`--resume` round-trips weights and
optimizer state.

Usage:
    python gt1030/test_optimizer_parity.py [--model DIR]
Default model: D:\deepsleep\tiny_models\ausimian__tiny-random-Qwen2ForCausalLM
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, 'igpu_train.py')
DEFAULT_MODEL = r'D:\deepsleep\tiny_models\ausimian__tiny-random-Qwen2ForCausalLM'
OPTIMIZERS = ['adamw', 'adamw8bit', 'muon']


def run(args):
    p = subprocess.run([sys.executable, KIT] + args, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    return p.returncode, (p.stdout or '') + (p.stderr or '')


def result_line(out):
    for line in out.splitlines():
        if line.startswith('RESULT '):
            return line
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--skip-igpu', action='store_true')
    args = ap.parse_args()
    if not os.path.isdir(args.model):
        print('SKIP: model dir not found:', args.model)
        return 0
    backends = ['cpu']
    if not args.skip_igpu:
        try:
            import pyopencl as cl  # noqa: F401
            cl.create_some_context(interactive=False)
            backends.append('igpu')
        except Exception as exc:  # noqa: BLE001
            print('SKIP igpu backend:', str(exc)[:100])
    failures = []
    for backend in backends:
        for opt in OPTIMIZERS:
            rc, out = run(['--model', args.model, '--backend', backend, '--optimizer', opt,
                           '--steps', '2', '--check'])
            ok = rc == 0 and ('CHECK: PASS' in out) and ('optimizer=%s' % opt) in result_line(out)
            print('[%s] %-8s %-9s %s' % ('PASS' if ok else 'FAIL', backend, opt,
                                         result_line(out)[:150] if result_line(out) else out[-200:]))
            if not ok:
                failures.append((backend, opt))
    # save/resume round-trip with muon on the iGPU path when available
    backend = backends[-1]
    tmp = tempfile.mkdtemp(prefix='igpu_opt_parity_')
    try:
        rc, out = run(['--model', args.model, '--backend', backend, '--optimizer', 'muon',
                       '--steps', '2', '--save', tmp])
        saved = os.path.exists(os.path.join(tmp, 'optimizer.pt')) and \
            os.path.exists(os.path.join(tmp, 'trainable.pt'))
        rc2, out2 = run(['--model', args.model, '--backend', backend, '--optimizer', 'muon',
                         '--steps', '2', '--resume', tmp, '--check'])
        resumed = 'resumed optimizer state' in out2
        ok = rc == 0 and saved and rc2 == 0 and resumed and 'CHECK: PASS' in out2
        print('[%s] %-8s save/resume  saved=%s resumed=%s' % ('PASS' if ok else 'FAIL',
                                                             backend, saved, resumed))
        if not ok:
            failures.append((backend, 'save/resume'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nOPTIMIZER PARITY TESTS %s' % ('PASSED' if not failures else 'FAILED'))
    if failures:
        print('failures:', failures)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
