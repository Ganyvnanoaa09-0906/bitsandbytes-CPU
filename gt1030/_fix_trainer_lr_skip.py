# -*- coding: utf-8 -*-
"""按行精确修 trainer：
   ① 真的用 lr_used（之前只把它写进日志，opt_step 收到的还是常数 args.lr —— 40k 那次实际上是常数 3e-5 跑崩的）
   ② 非有限 loss 跳过该 batch
   ③ watcher 子进程输出强制 UTF-8
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_train.py'
lines = io.open(p, encoding='utf-8').read().splitlines()
out = []
for ln in lines:
    if ln.strip() == 'gn = model.opt_step(args.lr, start_step + st)':
        out.append('        gn = model.opt_step(lr_used, start_step + st)   # 真的用调度后的 lr')
        continue
    if ln.strip() == 'loss = model.forward(tokens)':
        out.append(ln)
        out.append('        if not np.isfinite(loss):')
        out.append('            n_skip += 1')
        out.append('            msg0 = \'[igpu] step %d/%d SKIP (loss 非有限)\' % (st, args.steps)')
        out.append('            print(msg0, flush=True)')
        out.append('            if logf:')
        out.append('                logf.write(msg0 + "\\n"); logf.flush()')
        out.append('            continue')
        continue
    if ln.strip() == 'lr_used = args.lr':
        out.append('    n_skip = 0')
        out.append(ln)
        continue
    out.append(ln)
txt = '\n'.join(out) + '\n'
# 汇总里带上 skip 次数
txt = txt.replace("""    if logf:
        logf.write('[igpu] SUMMARY steps=%d median=%.3fs tok/s=%.0f loss %.4f -> %.4f\\n'""",
"""    if n_skip:
        print('[igpu] 跳过了 %d 个非有限 loss 的 batch' % n_skip, flush=True)
    if logf:
        logf.write('[igpu] SUMMARY steps=%d median=%.3fs tok/s=%.0f loss %.4f -> %.4f\\n'""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(txt)
print('trainer 修好：lr_used + skip 非有限 loss')

p3 = r'D:\work\bitsandbytes-CPU\gt1030\_igpu_40k_watch.py'
w = io.open(p3, encoding='utf-8').read()
w = w.replace("capture_output=True, text=True)", "capture_output=True, text=True, encoding='utf-8', errors='replace')")
w = w.replace("""                           check=True, capture_output=True)""",
              """                           check=True, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')""")
io.open(p3, 'w', encoding='utf-8', newline='\n').write(w)
print('watcher 编码修好')
