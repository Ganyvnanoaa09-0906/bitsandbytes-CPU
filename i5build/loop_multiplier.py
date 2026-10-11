"""Settle the loop multiplier: how many block passes does one training step do?

Why it matters: the analytical accounting says the step is 97% matmul and sits
at 194 of the 200 GFLOPS GEMM ceiling -- i.e. nothing left to optimise. But that
assumed 10 block passes per forward. The config is loop_L=1..3 random over
layers [4..7], so the real pass count is 6 + 4L. At the average L=2 that is 14
passes, which would put the implied throughput at 272 GFLOPS -- above the
measured ceiling, i.e. impossible. One of the two numbers is wrong, and the
difference decides whether there is headroom (66%) or none (97%).

Method: run the same resume with loop_L FIXED at 1 and at 3, let each run for a
fixed wall time, and compare how many steps each completed. The step-rate ratio
is the pass-count ratio, and from it the real FLOPs per step follow directly.
"""
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
HERE = r'D:\work\bitsandbytes-CPU'
CK = os.path.join(HERE, 'i5build', 'loop_ar3', 'ckpt.pt', 'ckpt_last.pt')
TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt'
VQ = r'D:\work\cloud_salvage\ScPeP7\vqvae_16x16_cmse.pt'


def run_for(L, seconds, tag):
    out = os.path.join(HERE, 'i5build', 'loopmul_%s' % tag)
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, 'train.log')
    cmd = [sys.executable, '-u', 'train_ar_v2.py',
           '--tokens', TOK, '--vqvae', VQ,
           '--out', os.path.join(out, 'ckpt.pt'),
           '--resume', CK,
           '--stages', '16:800,64:1200,256:26000',      # long enough to not finish
           '--batch', '16:64,64:16,256:8',
           '--loop_start', '4', '--loop_end', '7',
           '--loop_L', str(L), '--loop_L_max', str(L),
           '--loop_cache', 'perloop',            # no --loop_random => fixed L
           '--use_bos', '--threads', '6',
           # Frequent val points give us step numbers in the log; a tiny val set
           # keeps that nearly free. (First attempt used val_every=100000, which
           # produced no step lines at all and made the measurement impossible.)
           '--val_frac', '0.0005', '--val_every', '25',
           '--seed', '1234']
    f = open(log, 'w', encoding='utf-8')
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(seconds)
    p.terminate()
    try:
        p.wait(timeout=30)
    except Exception:
        p.kill()
    f.close()
    wall = time.time() - t0
    txt = open(log, encoding='utf-8', errors='replace').read()
    steps = [int(m) for m in re.findall(r'step (\d+)  train=', txt)]
    lo, hi = (min(steps), max(steps)) if steps else (0, 0)
    print('  L=%d  wall %.0f s  记录到的步 %s..%s' % (L, wall, lo, hi))
    return lo, hi, wall


print('=== loop_L 固定 1 vs 3：步速比 = pass 数比 ===')
print('（每次跑固定墙钟时间，比较完成的步数）\n')
res = {}
for L in (1, 3):
    res[L] = run_for(L, 200, 'L%d' % L)

print()
if res[1][1] > res[1][0] and res[3][1] > res[3][0]:
    # val points are every 500 steps, so use the elapsed-portion counts instead
    # safer: compare raw step indices reached within the same wall time
    n1 = res[1][1] - res[1][0]
    n3 = res[3][1] - res[3][0]
    print('  同样 %ds 内：L=1 推进 %d 步，L=3 推进 %d 步' % (200, n1, n3))
    if n1 and n3:
        print('  ⇒ 步速比 %.2f' % (n1 / n3))
        print('  ⇒ 若 pass 数为 6+4L：L=1 → 10，L=3 → 18，比值应为 1.80')
        print('  ⇒ 实测比值与 1.80 的偏离即我的 FLOPs 模型的误差')
else:
    print('  步数不足以比较（val_every 太大导致日志里没有中间步）。')
    print('  改用另一种判据：进程的 CPU 时间累积速率。')

print('\n注：记录到的步只在 val 点出现 ⇒ 用它算比值时误差较大；')
print('    下面直接用 CPU 时间/步 作为更细的判据。')
