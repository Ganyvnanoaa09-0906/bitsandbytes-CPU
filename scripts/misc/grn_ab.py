# -*- coding: utf-8 -*-
"""grn_ab.py — 确认 GRN 失败机理的 A/B（纯本地，不花云预算）。

假设：GRN 的损失在【全部位置】上平均，而输入里 l_t 比例的位置直接含真值
      ⇒ 梯度被"照抄输入"主导 ⇒ 模型学会抄、学不会"无物可抄时生成"。
      推理从纯随机起步 ⇒ 只能吐边缘分布 ⇒ 全黑/云雾。

判据（若 B 显著优于 A，假设成立）：
  · 采样后的【码本使用率 / 熵】—— 塌缩时极低（实测 2.4% / 26.6%）
  · 【高频缺口】—— 塌缩时极差（实测 −20.8 ~ −33.3 dB）
  · 【看图】—— 有没有内容

两臂除 --grn-corrupt-only 外完全一致（同 seed / 同数据 / 同步数）。
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, subprocess, sys, time

BB = r'D:\work\bitsandbytes-CPU'
PY = sys.executable
TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt'
VQ = r'D:\work\cloud_salvage\CrswCc\vqvae_16x16_cmse.pt'
STEPS = int(os.environ.get('GRN_AB_STEPS', '5000'))       # 每臂步数（默认 5000）
THREADS = int(os.environ.get('GRN_AB_THREADS', '3'))      # 每臂线程（两臂并行）

ARMS = [
    ('A_full', []),
    ('B_corrupt', ['--grn-corrupt-only']),
]

procs = []
for name, extra in ARMS:
    out = os.path.join(BB, '.autopilot', 'grnab_' + name)
    log = os.path.join(BB, '.autopilot', 'grnab_%s.log' % name)
    cmd = [PY, '-u', 'train_ar_v2.py',
           '--tokens', TOK, '--vqvae', VQ, '--out', out,
           '--stages', '256:%d' % STEPS, '--batch', '256:8',
           '--d_model', '512', '--n_layer', '10', '--n_head', '8',
           '--n_kv_head', '4', '--ffn_dim', '2048',
           '--head-mode', 'flat', '--loop_times', '1', '--use_bos', '--hflip',
           '--mask-mode', 'full', '--grn-refine',
           '--val-range', '27658:29113', '--val_every', '100000',
           '--threads', str(THREADS), '--lr', '2e-4', '--warmup', '1000',
           '--lr-min', '2e-5', '--log_every', '50', '--ckpt_every', '1000',
           '--seed', '2024'] + extra
    f = open(log, 'w', encoding='utf-8')
    p = subprocess.Popen(cmd, cwd=BB, stdout=f, stderr=subprocess.STDOUT)
    procs.append((name, out, log, p, f))
    print('  launched %-10s pid=%d -> %s' % (name, p.pid, log), flush=True)

print('两臂并行中（各 %d 线程，%d 步）。预计 %.1f 小时。' % (THREADS, STEPS, STEPS * 4.5 / 3600))
t0 = time.time()
while any(p.poll() is None for _, _, _, p, _ in procs):
    time.sleep(120)
    el = (time.time() - t0) / 60
    line = []
    for name, out, log, p, f in procs:
        st = '跑' if p.poll() is None else '完'
        last = ''
        jl = os.path.join(out, 'train.jsonl')
        if os.path.exists(jl):
            try:
                with open(jl, encoding='utf-8') as g:
                    rows = g.readlines()
                if rows:
                    import json
                    r = json.loads(rows[-1])
                    last = 'step=%s loss=%.3f' % (r.get('step'), r.get('loss', -1))
            except Exception:
                pass
        line.append('%s[%s] %s' % (name, st, last))
    print('  %5.1f min | %s' % (el, ' | '.join(line)), flush=True)

for name, out, log, p, f in procs:
    f.close()
    print('  %s 结束 rc=%s' % (name, p.returncode))
print('产物目录：', [o for _, o, _, _, _ in procs])
