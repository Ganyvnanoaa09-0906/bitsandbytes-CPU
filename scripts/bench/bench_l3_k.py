# -*- coding: utf-8 -*-
"""bench_l3_k.py — 扫 k（每键 embedding 数），找 L³ 的速度-容量折中最优点

k 越大：容量越大、gather 越大、einsum 越大
k 越小：更快，但每个键能缓存的信息更少

同时扫 1gram（键=当前 token，键空间 1024）vs 4gram（键=空间邻域，键空间大）
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
from scripts.lib.l3_ffn import L3FFN, causal_spatial_key

torch.set_num_threads(6)
D, G = 512, 16
T = G * G
B = 8


def bench(fn, iters=4, warm=2):
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


x = torch.randn(B, T, D)
tok = torch.randint(0, 1024, (B, T))
k1 = causal_spatial_key(tok, G, 4096, '1gram')
k4 = causal_spatial_key(tok, G, 4096, '4gram')

sw = SwiGLU(D, 2048)
o = torch.optim.AdamW(sw.parameters(), lr=1e-4)


def tr():
    l = sw(x).pow(2).mean()
    o.zero_grad(set_to_none=True); l.backward(); o.step()


base = bench(tr)
print('=' * 78)
print('L³ k/V/键模式 扫描  (B=%d T=%d d=%d)' % (B, T, D))
print('=' * 78)
print('  SwiGLU 基线  %.2f ms   1.00x   3.15M 参数' % (base * 1e3), flush=True)
print()
print('  %-34s %10s %9s %10s' % ('配置', '训练 ms', '相对', '单层参数'), flush=True)

for mode, kk in (('1gram', k1), ('4gram', k4)):
    for V in (1024, 4096):
        for k in (2, 4, 8, 16):
            m = L3FFN(D, table_size=V, k_emb=k, d_emb=128, d_up=128)
            o2 = torch.optim.AdamW(m.parameters(), lr=1e-4)

            def tr2():
                l = m(x, kk).pow(2).mean()
                o2.zero_grad(set_to_none=True); l.backward(); o2.step()

            try:
                dt = bench(tr2)
                np_ = sum(p.numel() for p in m.parameters())
                print('  %-34s %10.2f %8.2fx %9.2fM'
                      % ('%s V=%-5d k=%-3d' % (mode, V, k), dt * 1e3, base / dt, np_ / 1e6), flush=True)
            except Exception as e:
                print('  %-34s 失败: %s' % ('%s V=%d k=%d' % (mode, V, k), str(e)[:40]), flush=True)
            del m, o2

print()
print('  判据: 相对 > 1.00x 说明比 SwiGLU 快', flush=True)
print('DONE', flush=True)
