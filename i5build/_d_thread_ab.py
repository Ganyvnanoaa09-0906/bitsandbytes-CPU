import os, re, subprocess, sys, time, json, shutil
HERE = r'D:\work\bitsandbytes-CPU'
CK = os.path.join(HERE, 'i5build', 'loop_ar3', 'ckpt.pt', 'ckpt_last.pt')
TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt'
VQ = r'D:\work\cloud_salvage\ScPeP7\vqvae_16x16_cmse.pt'
sys.stdout.reconfigure(encoding='utf-8')

def battery():
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command',
                            '(Get-CimInstance Win32_Battery).BatteryStatus'],
                           capture_output=True, text=True, timeout=15)
        return {'1': '电池(放电)!', '2': '交流电'}.get(r.stdout.strip(), r.stdout.strip())
    except Exception:
        return '?'

def run(threads, L, seconds, tag):
    out = os.path.join(HERE, 'i5build', '_d_ab_%s' % tag)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, 'train.log')
    jl = os.path.join(out, 'ckpt.pt', 'train.jsonl')
    cmd = [sys.executable, '-u', 'train_ar_v2.py', '--tokens', TOK, '--vqvae', VQ,
           '--out', os.path.join(out, 'ckpt.pt'), '--resume', CK,
           '--stages', '16:800,64:1200,256:26000', '--batch', '16:64,64:16,256:8',
           '--loop_start', '4', '--loop_end', '7', '--loop_L', str(L), '--loop_L_max', str(L),
           '--loop_cache', 'perloop', '--use_bos', '--threads', str(threads),
           '--val_every', '1000000', '--log_every', '1', '--no-ckpt', '--seed', '1234']
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
    ss = []
    try:
        for line in open(jl, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('step_s'):
                ss.append((r['step'], r['step_s']))
    except Exception:
        pass
    body = [s for (st, s) in ss if st and st > 22010]
    body.sort()
    med = body[len(body) // 2] if body else float('nan')
    print('  [%s] threads=%d L=%d 墙钟%.0fs | 记录%d步 | 中位 step_s=%.3f  min=%.3f max=%.3f  (电源:%s)'
          % (tag, threads, L, time.time() - t0, len(ss), med,
             body[0] if body else -1, body[-1] if body else -1, battery()), flush=True)
    return med

print('电源:', battery())
print('=== 线程数 A/B（固定 L=1，同一 ckpt，交错两轮）===')
res = {}
for rnd in range(2):
    print('--- 轮 %d ---' % (rnd + 1))
    for th in (6, 5):
        m = run(th, 1, 100, 't%d_r%d' % (th, rnd))
        res.setdefault(th, []).append(m)
print('\n=== 汇总 ===')
for th in (6, 5):
    v = [x for x in res[th] if x == x]
    if v:
        print('  threads=%d 中位 step_s = %.3f  (%s)' % (th, sorted(v)[len(v)//2], v))
print('\n参考：micro 基准中 L=1 约 3.35s/步；历史 loop_ar3 均值 3.99s/步（含 random L 和其他状态）')
