# -*- coding: utf-8 -*-
"""bench_l3_local.py — L³ 替换 SwiGLU 到底值不值（本地快速迭代）

理论账（每 token）:
  SwiGLU 512→2048→512 (含 gate) :  6.3 MFLOP
  L³ (k=8, d_emb=128, d_up=128) :  0.70 MFLOP      ⇒ 少 9×

内存账（每层每步, b=8 T=256 = 2048 token）:
  SwiGLU: 权重 3.15M × 4B = 12.6 MB（跨 batch 摊销，只读一遍）
  L³    : gather 2048 × 8×(512+128) × 4B = 42 MB（每 token 各自查）

CPU 拐点 ≈ 221 GFLOPS / 19 GB/s = 11.6 FLOP/字节
  SwiGLU 强度 = 6.3e6 / 6.15e3 = 1024 FLOP/B   → 算力受限
  L³     强度 = 0.70e6 / 20.5e3 = 34 FLOP/B    → 也算力受限

⇒ 两个都在拐点之上，L³ 应该赢。但 gather 的实际效率是关键，必须实测。
"""
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_l3 import causal_spatial_key

torch.set_num_threads(6)


def bench(fn, iters=6, warm=2):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


class SwiGLU(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w3 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)

    def forward(self, x, keys=None):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class L3(nn.Module):
    """L³ 层，静态键由图外算好传进来（键不依赖 hidden state）"""

    def __init__(self, d, table_size=65536, k_emb=8, d_emb=128, d_up=128):
        super().__init__()
        self.k = k_emb
        self.WK = nn.Embedding(table_size, k_emb * d)
        self.WV = nn.Embedding(table_size, k_emb * d_emb)
        self.W_up = nn.Linear(d_emb, d_up)
        self.norm = nn.LayerNorm(d_up)
        self.W_mix = nn.Linear(d_up + d, d)
        nn.init.normal_(self.WK.weight, std=0.02)
        nn.init.normal_(self.WV.weight, std=0.02)

    def forward(self, x, keys):
        B, T, D = x.shape
        K = self.WK(keys).view(B, T, self.k, D)
        V = self.WV(keys).view(B, T, self.k, -1)
        s = torch.einsum('btkd,btd->btk', K, x)
        a = torch.softmax(s, dim=-1)
        z = torch.einsum('btk,btke->bte', a, V)
        u = self.norm(self.W_up(z))
        return self.W_mix(torch.cat([u, x], dim=-1))


D, G = 512, 16
T = G * G
print('=' * 74)
print('L³  vs  SwiGLU  —— 本地实测 (d=512, grid=16, threads=6)')
print('=' * 74)
print('  %-30s %6s %10s %10s %10s' % ('模块', 'B', '前向 ms', '训练 ms', '相对'), flush=True)

for B in (8, 32):
    x = torch.randn(B, T, D)
    tok = torch.randint(0, 1024, (B, T))
    key4 = causal_spatial_key(tok, G, 1024, 65536, '4gram')
    key1 = causal_spatial_key(tok, G, 1024, 65536, '1gram')

    sw = SwiGLU(D, 2048)
    o1 = torch.optim.AdamW(sw.parameters(), lr=1e-4)

    def tr_sw():
        y = sw(x)
        l = y.pow(2).mean()
        o1.zero_grad(set_to_none=True); l.backward(); o1.step()

    tf_sw = bench(lambda: sw(x))
    tt_sw = bench(tr_sw)
    print('  %-30s %6d %10.2f %10.2f %10s'
          % ('SwiGLU 512->2048->512', B, tf_sw * 1e3, tt_sw * 1e3, '1.00x'), flush=True)

    for tag, kk in (('L³ 1gram V=64K', key1), ('L³ 4gram V=64K', key4)):
        m = L3(D)
        o2 = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def tr_l3():
            y = m(x, kk)
            l = y.pow(2).mean()
            o2.zero_grad(set_to_none=True); l.backward(); o2.step()

        try:
            tf = bench(lambda: m(x, kk))
            tt = bench(tr_l3)
            npar = sum(p.numel() for p in m.parameters())
            act = m.k * D + m.k * 128 + 128 * 128 + 128 * (128 + D) + 2 * 128
            print('  %-30s %6d %10.2f %10.2f %9.2fx   (表%.1fM 活跃%.0fk)'
                  % (tag, B, tf * 1e3, tt * 1e3, tt_sw / tt, npar / 1e6, act / 1e3), flush=True)
        except Exception as e:
            print('  %-30s %6d 失败: %s' % (tag, B, str(e)[:50]), flush=True)
        del m, o2
    del sw, o1, x, tok, key4, key1

print()
print('  判据: 训练列 < 1.00x 说明 L³ 更快', flush=True)
print('DONE', flush=True)
