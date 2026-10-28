# -*- coding: utf-8 -*-
"""bench_l3_sparse.py — L³ 走稀疏梯度路径能不能救回来

上一测（bench_l3_local.py）结论：
  L³ 前向快 2.5×，但训练慢 10×（0.10x）
  根因：nn.Embedding 默认 sparse=False，反向产生【全表大小】的稠密梯度
        表 335.9M 参数 = 1.34 GB，每层每步要清零+散射 2.7 GB

本测对比三条路：
  A) SwiGLU 基线
  B) L³ + 稠密梯度（复现上一测）
  C) L³ + sparse=True + SparseAdam      ← 关键
  D) L³ + 小表 V=4096 + 稠密梯度         ← 另一条路：把表做小
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
import os, sys, time
import torch
import torch.nn
import torch.nn.functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.spatial_l3 import causal_spatial_key

torch.set_num_threads(6)


def bench(fn, iters=5, warm=2):
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
    def __init__(self, d, table_size, k_emb=8, d_emb=128, d_up=128, sparse=False):
        super().__init__()
        self.k = k_emb
        self.WK = nn.Embedding(table_size, k_emb * d, sparse=sparse)
        self.WV = nn.Embedding(table_size, k_emb * d_emb, sparse=sparse)
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
B = 8
x = torch.randn(B, T, D)
tok = torch.randint(0, 1024, (B, T))
keys = causal_spatial_key(tok, G, 1024, 65536, '4gram')

print('=' * 78)
print('L³ 稀疏梯度路径对比  (B=%d, T=%d, d=%d)' % (B, T, D))
print('=' * 78)
print('  %-34s %10s %10s %10s' % ('方案', '训练 ms', '相对', '参数量'), flush=True)

# ---- A) SwiGLU 基线 ----
sw = SwiGLU(D, 2048)
o = torch.optim.AdamW(sw.parameters(), lr=1e-4)


def tr_sw():
    l = sw(x).pow(2).mean()
    o.zero_grad(set_to_none=True); l.backward(); o.step()


tt_sw = bench(tr_sw)
n_sw = sum(p.numel() for p in sw.parameters())
print('  %-34s %10.2f %10s %9.2fM' % ('A) SwiGLU 512->2048->512', tt_sw * 1e3, '1.00x', n_sw / 1e6), flush=True)
del sw, o

# ---- B) L³ 稠密梯度 ----
m = L3(D, 65536, sparse=False)
o = torch.optim.AdamW(m.parameters(), lr=1e-4)


def tr_dense():
    l = m(x, keys).pow(2).mean()
    o.zero_grad(set_to_none=True); l.backward(); o.step()


tt = bench(tr_dense, iters=3)
print('  %-34s %10.2f %9.2fx %9.2fM' % ('B) L³ V=64K 稠密梯度', tt * 1e3, tt_sw / tt,
                                        sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
del m, o

# ---- C) L³ 稀疏梯度 ----
try:
    m = L3(D, 65536, sparse=True)
    emb_params = list(m.WK.parameters()) + list(m.WV.parameters())
    dense_params = [p for n, p in m.named_parameters() if not n.startswith(('WK', 'WV'))]
    o_sp = torch.optim.SparseAdam(emb_params, lr=1e-4)
    o_de = torch.optim.AdamW(dense_params, lr=1e-4)

    def tr_sparse():
        l = m(x, keys).pow(2).mean()
        o_sp.zero_grad(set_to_none=True); o_de.zero_grad(set_to_none=True)
        l.backward()
        o_sp.step(); o_de.step()

    tt = bench(tr_sparse, iters=5)
    print('  %-34s %10.2f %9.2fx %9.2fM' % ('C) L³ V=64K sparse+SparseAdam', tt * 1e3, tt_sw / tt,
                                            sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
    del m, o_sp, o_de
except Exception as e:
    print('  C) 失败: %s' % (str(e)[:70]), flush=True)

# ---- D) L³ 小表 + 稠密 ----
for V in (4096, 16384):
    m = L3(D, V, sparse=False)
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def tr_small():
        l = m(x, keys % V).pow(2).mean()
        o.zero_grad(set_to_none=True); l.backward(); o.step()

    tt = bench(tr_small, iters=4)
    print('  %-34s %10.2f %9.2fx %9.2fM' % ('D) L³ V=%d 稠密' % V, tt * 1e3, tt_sw / tt,
                                            sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
    del m, o

print()
print('  判据: 相对 > 1.00x 说明比 SwiGLU 慢', flush=True)
print('DONE', flush=True)
