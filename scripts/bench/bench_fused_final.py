# -*- coding: utf-8 -*-
"""bench_fused_final.py — 最终确认（多次重复 + 中位数 + 交叉验证）

上一轮单次测量噪声大（SwiGLU 单独 -2.2%，但加上 RMSNorm 却变成 +0.9%）。
本轮：每组跑 3 轮，取中位数；并做 A/B/A 交叉验证排除漂移。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import sys, time, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 84)
print('融合核最终确认（3 轮中位数 + A/B/A 交叉验证）')
print('=' * 84)

from scripts.lib import fused_cpu
from scripts.lib import small_image_model_v2 as M2

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
    for k, f in (('rms', M2.RMSNorm), ('swiglu', M2.SwiGLU), ('block', M2.BlockV2)):
        pass
    M2.RMSNorm.forward = ORIG['rms']
    M2.SwigGLU = getattr(M2, 'SwiGLU')   # no-op guard
    M2.SwiGLU.forward = ORIG['swiglu']
    M2.BlockV2.forward = ORIG['block']


def p_rms(self, x):
    return fused_cpu.fused_rmsnorm(x, self.w, None, self.eps)


def p_swiglu(self, x):
    a, b = self.w12(x).chunk(2, dim=-1)
    return self.w3(fused_cpu.fused_swiglu(a, b))


def p_block(self, x, pooled=None, ctx=None, cache=None, append=True, keys=None):
    """RMSNorm(fused) + 残差也走 fused_add_scale"""
    if self.cm == 'adaln' and pooled is not None:
        return ORIG['block'](self, x, pooled, ctx, cache, append, keys)
    h = self.ln1(x)
    x = fused_cpu.fused_add_scale(x, self.attn(h, cache=cache, append=append), self.ls1)
    x = fused_cpu.fused_add_scale(x, self.mlp(self.ln2(x)), self.ls2)
    return x


def measure(setup, n=6, warm=2):
    reset(); setup()
    torch.manual_seed(0)
    m = M2.SmallARImageModelV2(cfg)
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    m.train()
    for _ in range(warm):
        _, g = m.forward_logits(tok)
        g = g[0] if isinstance(g, (tuple, list)) else g
        l = F.cross_entropy(g.reshape(-1, 1024), tok.reshape(-1))
        o.zero_grad(); l.backward(); o.step()
    t0 = time.perf_counter()
    for _ in range(n):
        _, g = m.forward_logits(tok)
        g = g[0] if isinstance(g, (tuple, list)) else g
        l = F.cross_entropy(g.reshape(-1, 1024), tok.reshape(-1))
        o.zero_grad(); l.backward(); o.step()
    return (time.perf_counter() - t0) / n


CONFIGS = [
    ('eager（基线）', lambda: None),
    ('只 RMSNorm', lambda: setattr(M2.RMSNorm, 'forward', p_rms)),
    ('RMSNorm + SwiGLU', lambda: (setattr(M2.RMSNorm, 'forward', p_rms),
                                  setattr(M2.SwiGLU, 'forward', p_swiglu))),
    ('RMSNorm + SwiGLU + 残差', lambda: (setattr(M2.RMSNorm, 'forward', p_rms),
                                        setattr(M2.SwiGLU, 'forward', p_swiglu),
                                        setattr(M2.BlockV2, 'forward', p_block))),
]

print()
print('[1] 每组 3 轮，取中位数')
res = {}
for label, setup in CONFIGS:
    ts = [measure(setup) for _ in range(3)]
    med = statistics.median(ts)
    res[label] = med
    print('    %-26s 中位 %.3f s   (三轮: %s)'
          % (label, med, ' '.join('%.3f' % t for t in ts)))

base = res['eager（基线）']
print()
print('[2] 相对基线')
for label, med in res.items():
    if label == 'eager（基线）':
        continue
    print('    %-26s %+.1f%%   (%.3fx)' % (label, (base / med - 1) * 100, base / med))

print()
print('[3] A/B/A 交叉验证（排除时间漂移）')
a1 = measure(lambda: None)
b1 = measure(lambda: (setattr(M2.RMSNorm, 'forward', p_rms),
                      setattr(M2.SwiGLU, 'forward', p_swiglu),
                      setattr(M2.BlockV2, 'forward', p_block)))
a2 = measure(lambda: None)
print('    基线A1 %.3f   全融合B %.3f   基线A2 %.3f' % (a1, b1, a2))
print('    ⇒ 基线漂移 %.2f%%，全融合相对稳定基线 %+.1f%%'
      % (abs(a2 / a1 - 1) * 100, ((a1 + a2) / 2 / b1 - 1) * 100))

reset()
print()
print('=' * 84)
print('DONE')
