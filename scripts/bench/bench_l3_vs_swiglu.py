# -*- coding: utf-8 -*-
"""bench_l3_vs_swiglu.py — 决定性实验：L³ 在【训练】路径上到底比 SwiGLU 快还是慢。

分析预测（需要实测验证）：
  FLOPs:  SwiGLU 4.19 MFLOP/token  vs  L³(d_up=128) 0.68 MFLOP/token   → L³ 少 6.4×
  带宽:   SwiGLU 12.6MB 权重跨 batch 摊销  vs  L³ 每 token 各自 gather  → L³ 更差

训练是算力受限还是带宽受限，决定谁赢。实测为准。

测三种配置 × 两种 batch，看前向和训练（前向+反向）的墙钟。
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
import time
import torch
import torch.nn
import torch.nn.functional

torch.set_num_threads(6)


class SwiGLU(nn.Module):
    """当前模型用的 FFN：512 -> 2048 -> 512，含 gate"""
    def __init__(self, d, ffn):
        super().__init__()
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w3 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def bench(fn, iters=8, warm=3):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


D, G = 512, 16
T = G * G


def make_l3(table_size, k_emb, d_emb, d_up):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scripts.lib.spatial_l3 import SpatialL3
    return SpatialL3(D, G, vocab=1024, table_size=table_size,
                     k_emb=k_emb, d_emb=d_emb, d_up=d_up, key_mode='4gram')


print('=' * 74)
print('SwiGLU  vs  空间 L³  —— 训练路径实测 (d_model=512, grid=16)')
print('=' * 74)
print('  %-30s %8s %10s %10s' % ('模块', 'B', '前向 ms', '训练 ms'), flush=True)

for B in (4, 8):
    tok = torch.randint(0, 1024, (B, T))
    x = torch.randn(B, T, D)

    # ---- SwiGLU ----
    sw = SwiGLU(D, 2048)
    opt = torch.optim.AdamW(sw.parameters(), lr=1e-4)

    def fwd_sw():
        return sw(x)

    def tr_sw():
        y = sw(x)
        loss = y.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    tf = bench(fwd_sw)
    tt = bench(tr_sw)
    nsw = sum(p.numel() for p in sw.parameters())
    print('  %-30s %8d %10.2f %10.2f   (%.2fM 参数)'
          % ('SwiGLU 512->2048->512', B, tf * 1e3, tt * 1e3, nsw / 1e6), flush=True)

    # ---- L³ 三种配置 ----
    for (V, k, de, du) in ((65536, 8, 128, 128), (65536, 8, 128, 256), (65536, 16, 256, 512)):
        m = make_l3(V, k, de, du)
        opt2 = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def fwd_l3():
            return m(x, tok)

        def tr_l3():
            y = m(x, tok)
            loss = y.pow(2).mean()
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            opt2.step()

        try:
            tf2 = bench(fwd_l3)
            tt2 = bench(tr_l3)
            n2 = sum(p.numel() for p in m.parameters())
            act = k * D + k * de + de * du + du * (du + D) + 2 * du
            print('  %-30s %8d %10.2f %10.2f   (表%.1fM 活跃%.0fk)'
                  % ('L³ V=%d k=%d d_emb=%d d_up=%d' % (V, k, de, du), B,
                     tf2 * 1e3, tt2 * 1e3, n2 / 1e6, act / 1e3), flush=True)
        except Exception as e:
            print('  L³(V=%d,k=%d,de=%d,du=%d) 失败: %s' % (V, k, de, du, str(e)[:50]), flush=True)
        del m, opt2
    print()
    del sw, opt, x, tok

print('DONE', flush=True)
