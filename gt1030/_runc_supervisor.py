# -*- coding: utf-8 -*-
"""run C 的监督者（按 phone 那边定的策略）：

  A) 每个新 ckpt → 算 val（382 张）→ 写 <ckpt>.health.json → 追加进度笔记
     · 停止规则：连续 2 个 ckpt 的 val 都没低于 BEST(=5.1963) → 停训练（别烧 6 小时）
  B) 回滚规则（扫描日志最近 200 步）：
     · 连续 3 步非有限 SKIP  或  最近 200 步里 gnorm>100 的 spike ≥5 次
     → 停训练 → 选最近一个 health=ok 的 ckpt → lr×0.5、warmup 50 续跑，日志打 === ROLLBACK ===
  C) 磁盘：只保留最近 2 个 `.opt.npz`（每个 341MB），旧的删掉

    python gt1030/_runc_supervisor.py
"""
from __future__ import annotations
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

PY = sys.executable
BASE = r'D:\work\bitsandbytes-CPU'
GT = os.path.join(BASE, 'gt1030')
F = r'F:\weibo_cos'
OUTDIR = os.environ.get('RUNC_OUTDIR', os.path.join(F, 'ar_run_igpu_c'))
TRAINLOG = os.environ.get('RUNC_LOG', os.path.join(F, 'ar_igpu_runc.log'))
NOTE = os.environ.get('RUNC_NOTE', r'D:\work\collab\shared\IGPU_RUNC_PROGRESS.md')
SUPLOG = os.environ.get('RUNC_SUPLOG', os.path.join(F, 'ar_igpu_runc_supervisor.log'))
BEST_VAL = float(os.environ.get('RUNC_BEST', '5.1963'))
STOP_TOL = float(os.environ.get('RUNC_TOL', '0.010'))   # 只有 val > BEST+TOL 才算'没进步'
CKPT_START = os.environ.get('RUNC_START', os.path.join(F, 'ar_run_igpu40k', 'ckpt_25000.pt'))
TOTAL_STEPS = int(os.environ.get('RUNC_STEPS', '15000'))
LR0 = float(os.environ.get('RUNC_LR', '5e-6'))
SPIKE = 100.0


def log(msg):
    line = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with io.open(SUPLOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    with io.open(NOTE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def trainer_pids():
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "Where-Object { $_.CommandLine -like '*ar_v2_train.py*' } | "
                          "Select-Object -ExpandProperty ProcessId"],
                         capture_output=True, text=True, encoding='utf-8', errors='replace')
    return [int(x) for x in (out.stdout or '').split() if x.strip().isdigit()]


def launch(lr, start_ckpt, tag):
    cmd = [PY, os.path.join(GT, 'ar_v2_train.py'), '--ckpt', start_ckpt,
           '--steps', str(TOTAL_STEPS), '--lr', '%.3e' % lr, '--lr-min', '1e-6',
           '--warmup', '50' if tag == 'rollback' else os.environ.get('RUNC_WARMUP', '200'),
           '--spike', str(SPIKE),
           '--opt-state', '1', '--save-every', '2500', '--log', TRAINLOG,
           '--out-dir', OUTDIR, '--val-frac', '0.02', '--shuffle', '1']
    env = dict(os.environ, AR_ARENA_MB='3200')
    log('LAUNCH(%s) lr=%.2e ckpt=%s' % (tag, lr, os.path.basename(start_ckpt)))
    with io.open(SUPLOG, 'ab') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=f, cwd=BASE, env=env)
    return p.pid


def read_log():
    try:
        with io.open(TRAINLOG, encoding='utf-8', errors='replace') as f:
            return f.read().splitlines()
    except Exception:
        return []


def parse_line(ln):
    m = re.search(r'step (\d+)/%d.*?loss=([-\d.a-z]+).*?gnorm=([-\d.a-z]+)' % TOTAL_STEPS, ln)
    if not m:
        return None
    try:
        return int(m.group(1)), float(m.group(2)), float(m.group(3))
    except ValueError:
        return int(m.group(1)), None, None


def trip_reason(lines):
    """看最近 200 步：连续 3 步非有限 → 'skip'；gnorm>100 次数 ≥5 → 'spike'"""
    tail = [parse_line(x) for x in lines if 'step ' in x]
    tail = [t for t in tail if t][-200:]
    if not tail:
        return None
    consec, spike = 0, 0
    for _, loss, gn in tail:
        bad = (loss is None) or (gn is None)
        consec = consec + 1 if bad else 0
        if gn is not None and gn > SPIKE:
            spike += 1
    if consec >= 3:
        return 'skip-consec'
    if spike >= 5:
        return 'spike'
    return None


def val_of(ckpt):
    r = subprocess.run([PY, os.path.join(GT, 'ar_v2_eval.py'), '--ckpt', ckpt],
                       capture_output=True, text=True, encoding='utf-8', errors='replace',
                       env=dict(os.environ, AR_ARENA_MB='3200'))
    out = (r.stdout or '') + (r.stderr or '')
    m = re.search(r'val_loss=([\d.]+)', out)
    return float(m.group(1)) if m else None


def write_health(ckpt, val, ok=True):
    h = {'ckpt': os.path.basename(ckpt), 'val': val, 'health': 'ok' if ok else 'bad',
         'ts': time.strftime('%Y-%m-%d %H:%M:%S')}
    with io.open(ckpt + '.health.json', 'w', encoding='utf-8') as f:
        json.dump(h, f, ensure_ascii=False)
    return h


def prune_opt(keep=2):
    fs = [os.path.join(OUTDIR, x) for x in os.listdir(OUTDIR) if x.endswith('.opt.npz')]
    fs.sort(key=os.path.getmtime)
    for p in fs[:-keep]:
        try:
            os.remove(p); log('删除旧 opt 状态 %s' % os.path.basename(p))
        except Exception as e:
            log('删不掉 %s: %r' % (p, e))


def best_ok_ckpt():
    """最近一个 health=ok 的 ckpt（没有就退回 run A 的 ckpt_25000）"""
    cands = []
    for x in os.listdir(OUTDIR):
        if x.endswith('.health.json'):
            try:
                h = json.load(io.open(os.path.join(OUTDIR, x), encoding='utf-8'))
            except Exception:
                continue
            if h.get('health') == 'ok' and h.get('val') is not None:
                cands.append((h['ts'], os.path.join(OUTDIR, h['ckpt']), h['val']))
    cands.sort()
    return cands[-1] if cands else (None, CKPT_START, None)


def main():
    log('=== supervisor start: OUTDIR=%s BEST=%.4f ===' % (OUTDIR, BEST_VAL))
    seen = set()
    fails_in_row = 0
    global HIST
    HIST = []
    lr = LR0
    while True:
        pids = trainer_pids()
        lines = read_log()
        # B) 回滚判定
        why = trip_reason(lines)
        if why and pids:
            log('=== 触发回滚条件: %s ===' % why)
            for p in pids:
                subprocess.run(['taskkill', '/PID', str(p), '/F'], capture_output=True)
            time.sleep(5)
            _, ck, val = best_ok_ckpt()
            lr = max(lr * 0.5, 5e-7)
            log('=== ROLLBACK to %s (val=%s)，lr→%.2e ===' % (os.path.basename(ck), val, lr))
            launch(lr, ck, 'rollback')
            time.sleep(30)
            continue
        # A) 新 ckpt → 算 val
        if os.path.isdir(OUTDIR):
            for x in sorted(os.listdir(OUTDIR)):
                if not (x.startswith('ckpt_') and x.endswith('.pt')) or x in seen:
                    continue
                seen.add(x)
                ck = os.path.join(OUTDIR, x)
                if not os.path.exists(ck + '.opt.npz'):
                    time.sleep(20)          # 等 opt 状态写完
                val = val_of(ck)
                ok = val is not None and val < 99
                h = write_health(ck, val, ok)
                HIST.append({'ckpt': x, 'val': val})
                prune_opt(2)
                log('%s val=%s health=%s' % (x, val, h['health']))
                if val is not None and val < BEST_VAL:
                    fails_in_row = 0
                    log('   >>> 破纪录（< %.4f）' % BEST_VAL)
                elif val is not None and val <= BEST_VAL + STOP_TOL:
                    fails_in_row = 0
                    log('   在噪声内（%.4f vs BEST %.4f + tol %.3f）→ 不算失败，继续'
                        % (val, BEST_VAL, STOP_TOL))
                else:
                    fails_in_row += 1
                    log('   明显未破（%.4f > %.4f，连续 %d 次）'
                        % (val, BEST_VAL + STOP_TOL, fails_in_row))
                    if fails_in_row >= 2 and pids:
                        prev = [h.get('val') for h in HIST[-2:]]
                        log('=== stopped by plateau rule: %s ===' % prev)
                        for p in pids:
                            subprocess.run(['taskkill', '/PID', str(p), '/F'], capture_output=True)
                        log('=== 已停止；最优仍是 ckpt_30000 val=%.4f ===' % BEST_VAL)
                        return
        if not pids:
            log('训练进程已退出，监督结束')
            return
        time.sleep(120)


if __name__ == '__main__':
    main()
