# -*- coding: utf-8 -*-
"""profile_train_gap.py — 找出训练步 150 GFLOPS vs sgemm 255 GFLOPS 的 1.7x 差距

四个待验证假设：
  H1  attention 的 K=head_dim=64 太小 ⇒ GEMM 效率低（K 小则 FMA 流水打不满）
  H2  非 GEMM 算子（norm/softmax/elementwise）占比高
  H3  GEMM 的 M 维（batch×T）不够大
  H4  内存带宽成了限制（权重 122 MB/步）

方法：真实模型跑一个训练步，用 profiler 按算子分类，
      并单独测「同形状纯 GEMM」的理论时间，对比求效率。
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
import sys, time, math
import torch
import torch.nn

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

THREADS = 6
torch.set_num_threads(THREADS)
print('=' * 80)
print('训练步的效率差距定位')
print('=' * 80)

B, T = 8, 256
cfg = SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=T, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
m = SmallARImageModelV2(cfg)
m.train()
tok = torch.randint(0, 1024, (B, T))
nparam = sum(p.numel() for p in m.parameters())

print()
print('[0] 规模与理论 FLOPs')
fwd_flops = 2 * nparam * B * T
train_flops = fwd_flops * 3
print('    参数 %.2fM   batch=%d  T=%d' % (nparam / 1e6, B, T))
print('    前向 FLOPs   %.1f GFLOP' % (fwd_flops / 1e9))
print('    训练 FLOPs   %.1f GFLOP（前向+反向+重算）' % (train_flops / 1e9))

# ---- 预热 + 计时 ----
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
for _ in range(2):
    t_, g_ = m.forward_logits(tok)
    g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
    loss = torch.nn.functional.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

N = 5
t0 = time.perf_counter()
for _ in range(N):
    t_, g_ = m.forward_logits(tok)
    g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
    loss = torch.nn.functional.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
step_s = (time.perf_counter() - t0) / N
gflops = train_flops / step_s / 1e9
print()
print('[1] 实测')
print('    步时        %.3f s' % step_s)
print('    有效算力    【%.1f GFLOPS】' % gflops)
print('    短时 sgemm 上限 255 GFLOPS ⇒ 效率 【%.0f%%】' % (gflops / 255 * 100))
print('    ⇒ 差距 %.2fx' % (255 / gflops))

# ---- profiler 分类 ----
print()
print('[2] 算子分类（torch.profiler）')
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(3):
        t_, g_ = m.forward_logits(tok)
        g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
        loss = torch.nn.functional.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

rows = []
for e in prof.key_averages():
    if e.self_device_time_total == 0 and e.self_cpu_time_total == 0:
        continue
    rows.append((e.key, e.self_cpu_time_total / 1000.0, e.count, e.cpu_time_total / 1000.0))
rows.sort(key=lambda x: -x[1])
tot = sum(r[1] for r in rows)
print('    profiler 总计时 %.0f ms（3 步）' % tot)
print()
print('    %-42s %10s %8s %8s' % ('算子', 'self ms', '次数', '占比'))
print('    ' + '-' * 72)
for k, self_ms, cnt, tot_ms in rows[:22]:
    print('    %-42s %10.2f %8d %7.1f%%' % (k[:42], self_ms, cnt, 100 * self_ms / tot))

# ---- 归类 ----
cat = {'GEMM/matmul': 0.0, 'attention(sdpa)': 0.0, 'elementwise': 0.0,
       'norm': 0.0, 'softmax': 0.0, 'copy/cat': 0.0, 'embedding': 0.0, 'other': 0.0}
for k, self_ms, cnt, tot_ms in rows:
    kl = k.lower()
    if 'mm' in kl or 'addmm' in kl or 'linear' in kl or 'matmul' in kl or 'bmm' in kl:
        cat['GEMM/matmul'] += self_ms
    elif 'sdpa' in kl or 'attention' in kl or 'scaled_dot' in kl:
        cat['attention(sdpa)'] += self_ms
    elif 'softmax' in kl:
        cat['softmax'] += self_ms
    elif 'norm' in kl or 'layer_norm' in kl or 'rms' in kl:
        cat['norm'] += self_ms
    elif 'copy' in kl or 'cat' in kl or 'clone' in kl or 'contiguous' in kl:
        cat['copy/cat'] += self_ms
    elif 'embedding' in kl or 'index' in kl:
        cat['embedding'] += self_ms
    elif any(x in kl for x in ('mul', 'add', 'silu', 'gelu', 'neg', 'exp', 'div',
                               'sub', 'pow', 'clamp', 'sigmoid', 'tanh', 'fill')):
        cat['elementwise'] += self_ms
    else:
        cat['other'] += self_ms
print()
print('    === 按类别 ===')
for k, v in sorted(cat.items(), key=lambda x: -x[1]):
    if v > 0:
        print('    %-20s %8.1f ms  %5.1f%%' % (k, v, 100 * v / tot))

# ---- H1: attention 的 K 维影响 ----
print()
print('[3] H1 验证：GEMM 的 K 维（head_dim）对效率的影响')
print('    %-34s %10s %14s' % ('形状 (M,K,N)', 'ms', 'GFLOPS'))
print('    ' + '-' * 62)
M = B * T
for K, N in ((64, 64), (64, 256), (128, 128), (128, 512), (256, 256), (512, 512),
             (512, 2048), (2048, 2048)):
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    for _ in range(2):
        _ = a @ b
    t0 = time.perf_counter()
    for _ in range(5):
        _ = a @ b
    t = (time.perf_counter() - t0) / 5
    gf = 2 * M * K * N / t / 1e9
    print('    M=%-5d K=%-5d N=%-5d %10.3f %14.1f' % (M, K, N, t * 1e3, gf))

print()
print('    ⇒ 若 K=64 的效率远低于 K=512，则【增大 head_dim】是训练加速手段')

print()
print('=' * 80)
print('DONE')
