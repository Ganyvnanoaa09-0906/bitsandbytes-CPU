# -*- coding: utf-8 -*-
"""两条改进：

① ar_v2_eval.py 顺带报噪声：用每个 batch 的 loss 算标准误（SEM = std/sqrt(n_batch)），
   这样"val 5.1963 vs 5.2308 到底差不差得动"当场就能判断，不用额外跑一遍 GPU。
② supervisor 的停止规则改成带容差：连续 2 个 ckpt 的 val 都 > BEST+0.010 才停，
   并在 progress 里记 'stopped by plateau rule: a / b'。
"""
import io

# ---------- ① eval 报 SEM ----------
p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_eval.py'
s = io.open(p, encoding='utf-8').read()
old = """    tot, cnt = 0.0, 0
    for i in range(0, len(va) - MM.B + 1, MM.B):
        blk = np.ascontiguousarray(va[i:i + MM.B])
        loss = model.forward(blk)
        if not np.isfinite(loss):
            print('  val 出现非有限 loss，停止')
            break
        tot += float(loss) * len(blk)
        cnt += len(blk)
    print('  val_loss=%.4f (%d 张)' % (tot / max(cnt, 1), cnt))"""
new = """    tot, cnt, per_batch = 0.0, 0, []
    for i in range(0, len(va) - MM.B + 1, MM.B):
        blk = np.ascontiguousarray(va[i:i + MM.B])
        loss = model.forward(blk)
        if not np.isfinite(loss):
            print('  val 出现非有限 loss，停止')
            break
        tot += float(loss) * len(blk)
        cnt += len(blk)
        per_batch.append(float(loss))
    mean = tot / max(cnt, 1)
    if len(per_batch) > 2:
        sd = float(np.std(per_batch, ddof=1))
        sem = sd / (len(per_batch) ** 0.5)
        print('  val_loss=%.4f ± %.4f (SEM, %d 张, %d 个 batch, batch 间 sd=%.3f)'
              % (mean, sem, cnt, len(per_batch), sd))
    else:
        print('  val_loss=%.4f (%d 张)' % (mean, cnt))"""
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('① eval 现在报 SEM')

# ---------- ② supervisor 容差 ----------
p2 = r'D:\work\bitsandbytes-CPU\gt1030\_runc_supervisor.py'
t = io.open(p2, encoding='utf-8').read()
t = t.replace("BEST_VAL = float(os.environ.get('RUNC_BEST', '5.1963'))",
              "BEST_VAL = float(os.environ.get('RUNC_BEST', '5.1963'))\n"
              "STOP_TOL = float(os.environ.get('RUNC_TOL', '0.010'))   # 只有 val > BEST+TOL 才算'没进步'")
t = t.replace("""                if val is not None and val < BEST_VAL:
                    fails_in_row = 0
                    log('   >>> 破纪录（< %.4f）' % BEST_VAL)
                else:
                    fails_in_row += 1
                    log('   未破 %.4f（连续 %d 次）' % (BEST_VAL, fails_in_row))
                    if fails_in_row >= 2 and pids:
                        log('=== 连续 2 个 ckpt 未破纪录 → 停止训练（省时间）===')
                        for p in pids:
                            subprocess.run(['taskkill', '/PID', str(p), '/F'], capture_output=True)
                        log('=== 已停止；最优仍是 ckpt_30000 val=%.4f ===' % BEST_VAL)
                        return""",
"""                if val is not None and val < BEST_VAL:
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
                        return""")
t = t.replace("""    seen = set()
    fails_in_row = 0""", """    seen = set()
    fails_in_row = 0
    global HIST
    HIST = []""")
t = t.replace("""                h = write_health(ck, val, ok)""",
              """                h = write_health(ck, val, ok)
                HIST.append({'ckpt': x, 'val': val})""")
io.open(p2, 'w', encoding='utf-8', newline='\n').write(t)
import py_compile
py_compile.compile(p2, doraise=True)
print('② supervisor 停止规则加容差（BEST+0.010），compile ok')
