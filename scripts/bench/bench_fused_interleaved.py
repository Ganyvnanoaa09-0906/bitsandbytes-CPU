# -*- coding: utf-8 -*-
"""bench_fused_interleaved.py — 交替测量（唯一能得到可信数字的方法）

今天第 5 个测量陷阱：本机训练速度有 ~5% 的时间漂移，
顺序测量（A组 → B组）会把漂移算成"组的差异"。
唯一可靠的做法是【交替】A/B/A/B/...，然后比较配对的比值。

本机速度漂移的已知来源：15W TDP + 温度墙 + turbo 状态。
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import sys, time, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 84)
print('交替测量：融合核的真实收益')
print('=' * 84)

from scripts.lib import fused_cpu
from scripts.lib import small_image_model_v2

B, T = 8, 256
cfg = M2.SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=T, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
tok = torch.randint(0, 1024, (B, T))
ORIG = {'rms': M2.RMSNorm.forward, 'swiglu': M2.SwiGLU.forward, 'block': M2.BlockV2.forward}


def reset():
    M2.RMSNorm.forward = ORIG['rms']
    M2.SwiGLU.forward = ORIG['swiglu']
    M2.BlockV2.forward = ORIG['block']


def p_rms(self, x):
    return fused_cpu.fused_rmsnorm(x, self.w, None, self.eps)


def p_swiglu(self, x):
    a, b = self.w12(x).chunk(2, dim=-1)
    return self.w3(fused_cpu.fused_swiglu(a, b))


def p_block(self, x, pooled=None, ctx=None, cache=None, append=True, keys=None):
    if self.cm == 'adaln' and pooled is not None:
        return ORIG['block'](self, x, pooled, ctx, cache, append, keys)
    h = self.ln1(x)
    x = fused_cpu.fused_add_scale(x, self.attn(h, cache=cache, append=append), self.ls1)
    x = fused_cpu.fused_add_scale(x, self.mlp(self.ln2(x)), self.ls2)
    return x


FULL = lambda: (setattr(M2.RMSNorm, 'forward', p_rms),
                setattr(M2.SwiGLU, 'forward', p_swiglu),
                setattr(M2.BlockV2, 'forward', p_block))
RMSONLY = lambda: setattr(M2.RMSNorm, 'forward', p_rms)
NONE = lambda: None

# 模型只建一次，权重固定；测的是【步时】，不涉及权重更新差异
BENCH = {}


def build():
    torch.manual_seed(0)
    m = M2.SmallARImageModelV2(cfg)
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    return m, o


m, o = build()


def step(n=4):
    m.train()
    t0 = time.perf_counter()
    for _ in range(n):
        _, g = m.forward_logits(tok)
        g = g[0] if isinstance(g, (tuple, list)) else g
        l = F.cross_entropy(g.reshape(-1, 1024), tok.reshape(-1))
        o.zero_grad(); l.backward(); o.step()
    return (time.perf_counter() - t0) / n


def one(setup, n=4, warm=1):
    reset(); setup()
    for _ in range(warm):
        step(1)
    return step(n)


print()
print('[1] 交替测量 6 轮（A=基线  B=只 RMSNorm  C=全融合）')
print('    轮次    A基线    B仅RMS   C全融合   B/A      C/A')
print('    ' + '-' * 56)
A_, B_, C_ = [], [], []
for i in range(6):
    a = one(NONE)
    b = one(RMSONLY)
    c = one(FULL)
    A_.append(a); B_.append(b); C_.append(c)
    print('    %-6d %7.3f  %7.3f  %7.3f  %6.2fx  %6.2fx'
          % (i + 1, a, b, c, a / b, a / c))

mA, mB, mC = statistics.median(A_), statistics.median(B_), statistics.median(C_)
print()
print('    中位数:  A %.3f   B %.3f   C %.3f' % (mA, mB, mC))
print('    ⇒ 只 RMSNorm  提速 【%.1f%%】(%.3fx)' % ((mA / mB - 1) * 100, mA / mB))
print('    ⇒ 全融合      提速 【%.1f%%】(%.3fx)' % ((mA / mC - 1) * 100, mA / mC))

# 配对比值（排除漂移）
rb = [a / b for a, b in zip(A_, B_)]
rc = [a / c for a, c in zip(A_, C_)]
print()
print('    [配对比值] 只RMSNorm: 中位 %.3fx  最小 %.3fx  最大 %.3fx'
      % (statistics.median(rb), min(rb), max(rb)))
print('    [配对比值] 全融合:    中位 %.3fx  最小 %.3fx  最大 %.3fx'
      % (statistics.median(rc), min(rc), max(rc)))
print('    ⇒ 配对比值排除了时间漂移，是可信数字')

print()
print('[2] 漂移幅度量化（看基线自身的波动）')
print('    A 序列: %s' % ' '.join('%.3f' % x for x in A_))
print('    A 波动 %.1f%%（min %.3f max %.3f）'
      % ((max(A_) / min(A_) - 1) * 100, min(A_), max(A_)))

reset()
print()
print('=' * 84)
print('DONE')
