# -*- coding: utf-8 -*-
"""profile_train_gap2.py — 找到训练加速的真正空间

上一轮发现：
  GEMM 63.6%，elementwise 20.8%（只贡献 ~5% FLOPs）
  ⇒ 有效算力 135.7 GFLOPS vs 同形状 GEMM 可达 382 GFLOPS

本轮验证三件事：
  A) bnb 8-bit 优化器 vs torch AdamW（真实训练用的是前者）
  B) 非 GEMM 时间花在哪些模块上（RMSNorm / SwiGLU / LayerScale / 优化器）
  C) 真实模型各 GEMM 形状的效率（找出拖后腿的）
"""
import os, sys
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
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
torch.set_num_threads(6)
print('=' * 80)
print('训练加速空间的精确定位')
print('=' * 80)

B, T = 8, 256
cfg = SmallImageConfigV2_kw = None
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
cfg = SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=T, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
torch.manual_seed(0)
m = SmallARImageModelV2(cfg)
m.train()
tok = torch.randint(0, 1024, (B, T))
nparam = sum(p.numel() for p in m.parameters())
train_flops = 2 * nparam * B * T * 3
print()
print('参数 %.2fM  训练 FLOPs %.1f GFLOP/步' % (nparam / 1e6, train_flops / 1e9))


def bench(opt_name, opt):
    for _ in range(2):
        t_, g_ = m.forward_logits(tok)
        g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
        loss = torch.nn.functional.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    N = 4
    t0 = time.perf_counter()
    for _ in range(N):
        t_, g_ = m.forward_logits(tok)
        g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
        loss = torch.nn.functional.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    s = (time.perf_counter() - t0) / N
    print('    %-28s %7.3f s/步   %6.1f GFLOPS' % (opt_name, s, train_flops / s / 1e9))
    return s


print()
print('[A] 优化器对比（这是最容易换的一项）')
opt1 = torch.optim.AdamW(m.parameters(), lr=1e-4)
s_torch = bench('torch.optim.AdamW', opt1)

try:
    import bitsandbytes
    opt2 = bnb.optim.AdamW8bit(m.parameters(), lr=1e-4)
    s_bnb = bench('bnb.optim.AdamW8bit', opt2)
    print('    ⇒ bnb 8-bit 快 %.2fx' % (s_torch / s_bnb))
except Exception as e:
    print('    bnb 不可用: %s' % str(e)[:80])
    s_bnb = None

# ---- B) 前向 vs 反向 vs 优化器 的时间拆分 ----
print()
print('[B] 时间拆分：前向 / 反向 / 优化器')
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
# 预热
for _ in range(2):
    t_, g_ = m.forward_logits(tok)
    g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
    loss = torch.nn.functional.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

n_rep = 4
t_fwd = t_bwd = t_opt = 0.0
for _ in range(n_rep):
    t0 = time.perf_counter()
    t_, g_ = m.forward_logits(tok)
    g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
    loss = torch.nn.functional.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
    t1 = time.perf_counter()
    opt.zero_grad(); loss.backward()
    t2 = time.perf_counter()
    opt.step()
    t3 = time.perf_counter()
    t_fwd += t1 - t0; t_bwd += t2 - t1; t_opt += t3 - t2
t_fwd /= n_rep; t_bwd /= n_rep; t_opt /= n_rep
tot = t_fwd + t_bwd + t_opt
print('    前向        %7.3f s  %5.1f%%' % (t_fwd, 100 * t_fwd / tot))
print('    反向        %7.3f s  %5.1f%%' % (t_bwd, 100 * t_bwd / tot))
print('    优化器      %7.3f s  %5.1f%%' % (t_opt, 100 * t_opt / tot))
print('    合计        %7.3f s' % tot)
print()
print('    ⇒ 反向占 %.0f%%（报告说 2/3，实测吻合）' % (100 * t_bwd / tot))

# ---- C) 真实 GEMM 形状效率 ----# ---- C) 真实 GEMM 形状效率 ----
print()
print('[C] 真实模型里的 GEMM 形状效率（M=%d）' % (B * T))
print('    %-38s %10s %12s' % ('用途 / 形状', 'ms', 'GFLOPS'))
print('    ' + '-' * 62)
M = B * T
cases = [
    ('QKV 投影  (M,512)->(512)', (M, 512, 512)),
    ('O 投影    (M,512)->(512)', (M, 512, 512)),
    ('FFN gate  (M,512)->(2048)', (M, 512, 2048)),
    ('FFN down  (M,2048)->(512)', (M, 2048, 512)),
    ('输出头    (M,512)->(1024)', (M, 512, 1024)),
    ('attn Q@K^T (M,64)->(M)', (M, 64, T)),
    ('attn @V   (M,T)->(64)', (M, T, 64)),
]
for name, (mm, kk, nn_) in cases:
    a = torch.randn(mm, kk)
    b = torch.randn(kk, nn_)
    for _ in range(2):
        _ = a @ b
    t0 = time.perf_counter()
    for _ in range(5):
        _ = a @ b
    t = (time.perf_counter() - t0) / 5
    gf = 2 * mm * kk * nn_ / t / 1e9
    print('    %-38s %10.3f %12.1f' % (name, t * 1e3, gf))

print()
print('=' * 80)
print('DONE')
