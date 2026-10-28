# -*- coding: utf-8 -*-
"""bench_fused_e2e2.py — 融合核端到端收益 + 正确的数值等价性验证

关键修正：数值对比必须用【同一份权重、只做前向】。
         上一版让两个模型各自被优化器更新了 7 步，权重已经不同。
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
import sys, time
import torch
import torch.nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 82)
print('融合核端到端收益（修正版）')
print('=' * 82)

from scripts.lib import fused_cpu
from scripts.lib import small_image_model_v2

B, T = 8, 256
cfg = M2.SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=T, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
torch.manual_seed(0)
model = M2.SmallARImageModelV2(cfg)
tok = torch.randint(0, 1024, (B, T))
nparam = sum(p.numel() for p in model.parameters())
flops = 2 * nparam * B * T * 3

# ============ 1. 同一权重下的数值等价性（这是唯一的正确做法）============
print()
print('[1] 数值等价性（同一份权重，只做前向）')
model.eval()
with torch.no_grad():
    _, g_ref = model.forward_logits(tok)
g_ref = g_ref[0] if isinstance(g_ref, (tuple, list)) else g_ref

orig_rms_forward = M2.RMSNorm.forward
orig_swiglu_forward = M2.SwiGLU.forward
orig_block_forward = M2.BlockV2.forward


def patched_rms_forward(self, x):
    return fused_cpu.fused_rmsnorm(x, self.w, None, self.eps)


def patched_swiglu_forward(self, x):
    a, b = self.w12(x).chunk(2, dim=-1)
    return self.w3(fused_cpu.fused_swiglu(a, b))


def patched_block_forward(self, x, pooled=None, ctx=None, cache=None, append=True, keys=None):
    if self.cm == 'adaln' and pooled is not None:
        return orig_block_forward(self, x, pooled, ctx, cache, append, keys)
    h = self.ln1(x)
    x = fused_cpu.fused_add_scale(x, self.attn(h, cache=cache, append=append), self.ls1)
    x = fused_cpu.fused_add_scale(x, self.mlp(self.ln2(x)), self.ls2)
    return x


M2.RMSNorm.forward = patched_rms_forward
M2.SwiGLU.forward = patched_swiglu_forward
M2.BlockV2.forward = patched_block_forward

model.eval()
with torch.no_grad():
    _, g_fus = model.forward_logits(tok)
g_fus = g_fus[0] if isinstance(g_fus, (tuple, list)) else g_fus
d = (g_ref - g_fus).abs().max().item()
sc = g_ref.abs().max().item()
print('    logits 最大绝对差 %.3e   相对 %.3e   %s'
      % (d, d / sc, 'OK' if d / sc < 1e-4 else 'FAIL'))

# 逐层定位（找出哪个 patch 引入误差）
print()
print('    逐层差分（只 patch 一个，看误差变化）')
M2.RMSNorm.forward = orig_rms_forward
M2.SWIGLU_forward = orig_swiglu_forward
M2.BlockV2.forward = orig_block_forward

for label, apply_fn in (
    ('只 patch RMSNorm', lambda: (setattr(M2.RMSNorm, 'forward', patched_rms_forward),)),
    ('只 patch SwiGLU', lambda: (setattr(M2.SwiGLU, 'forward', patched_swiglu_forward),)),
    ('只 patch Block 残差', lambda: (setattr(M2.BlockV2, 'forward', patched_block_forward),)),
):
    M2.RMSNorm.forward = orig_rms_forward
    M2.SwiGLU.forward = orig_swiglu_forward
    M2.BlockV2.forward = orig_block_forward
    apply_fn()
    model.eval()
    with torch.no_grad():
        _, gx = model.forward_logits(tok)
    gx = gx[0] if isinstance(gx, (tuple, list)) else gx
    dd = (gx - g_ref).abs().max().item()
    print('    %-22s 最大绝对差 %.3e  相对 %.3e  %s'
          % (label, dd, dd / sc, 'OK' if dd / sc < 1e-4 else 'FAIL'))

# ============ 2. 全 patch 下测训练步 ============
M2.RMSNorm.forward = orig_rms_forward
M2.SwiGLU.forward = orig_swiglu_forward
M2.BlockV2.forward = orig_block_forward


def run_steps(m, opt, n=5, warm=2, seed=0):
    torch.manual_seed(seed)
    m.train()
    for _ in range(warm):
        t_, g_ = m.forward_logits(tok)
        g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
        loss = F.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(n):
        t_, g_ = m.forward_logits(tok)
        g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
        loss = F.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return (time.perf_counter() - t0) / n


print()
print('[2] 训练步时（同一初始权重，各自建优化器）')
torch.manual_seed(0)
m1 = M2.SmallARImageModelV2(cfg)
o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4)
t_base = run_steps(m1, o1)
print('    基线 eager      %.3f s/步   %.1f GFLOPS' % (t_base, flops / t_base / 1e9))

M2.RMSNorm.forward = patched_rms_forward
M2.SwiGLU.forward = patched_swiglu_forward
M2.BlockV2.forward = patched_block_forward

torch.manual_seed(0)
m2 = M2.SmallARImageModelV2(cfg)
o2 = torch.optim.AdamW(m2.parameters(), lr=1e-4)
t_fus = run_steps(m2, o2)
print('    融合核          %.3f s/步   %.1f GFLOPS' % (t_fus, flops / t_fus / 1e9))

print()
print('    ⇒ 【提速 %.1f%%】 (%.3fx)' % ((t_base / t_fus - 1) * 100, t_base / t_fus))

# 消融：单个 patch 的贡献
print()
print('[3] 消融：每个 patch 单独贡献多少')
for label, setup in (
    ('RMSNorm', lambda: setattr(M2.RMSNorm, 'forward', patched_rms_forward)),
    ('SwiGLU', lambda: setattr(M2.SwiGLU, 'forward', patched_swiglu_forward)),
    ('Block 残差', lambda: setattr(M2.BlockV2, 'forward', patched_block_forward)),
):
    M2.RMSNorm.forward = orig_rms_forward
    M2.SwiGLU.forward = orig_swiglu_forward
    M2.BlockV2.forward = orig_block_forward
    setup()
    torch.manual_seed(0)
    mx = M2.SmallARImageModelV2(cfg)
    ox = torch.optim.AdamW(mx.parameters(), lr=1e-4)
    tx = run_steps(mx, ox)
    print('    %-14s %.3f s/步  ⇒  %.1f%%' % (label, tx, (t_base / tx - 1) * 100))

M2.RMSNorm.forward = orig_rms_forward
M2.SwiGLU.forward = orig_swiglu_forward
M2.BlockV2.forward = orig_block_forward

print()
print('=' * 82)
print('DONE')
