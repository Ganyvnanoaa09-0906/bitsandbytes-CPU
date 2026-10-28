# -*- coding: utf-8 -*-
"""l3_ffn.py — 生产版 L³ 层（替换 SwiGLU FFN）

论文: L³: Large Lookup Layers (ICML 2026, arXiv:2601.21461)
    L³(x, t) = W_mix · [ LayerNorm(W_up(V_t^T · Softmax(K_t · x))) ; x ]

与文本版的区别 —— 静态路由键用【因果空间邻域】而不是 LZW 后缀：
光栅顺序下位置 (r,c) 的因果邻域（生成时已知）：
        (r-1,c-1) (r-1,c) (r-1,c+1)
        (r,c-1)   [r,c]
所以键是 token 的函数，不依赖 hidden state，可预取。

【本机实测的关键结论】(bench_l3_local.py / bench_l3_sparse.py, B=8 T=256 d=512)
    SwiGLU 512→2048→512          260.06 ms   1.00x   3.15M 参数
    L³ V=64K 稠密梯度           3551.33 ms   0.07x 335.89M   ← 灾难
    L³ V=64K sparse+SparseAdam   317.25 ms   0.82x 335.89M
    L³ V=4096 稠密               【230.06 ms   1.13x  21.32M】  ★ 最优
    L³ V=16384 稠密              822.80 ms   0.32x  84.23M

根因：nn.Embedding 默认 sparse=False，反向分配【全表大小】的稠密梯度。
      V=64K 时表 335.9M 参数 = 1.34 GB，每层每步多 2.7 GB 访存。
      表做小到 V=4096（21M 参数）后稠密梯度可接受，且比 SwiGLU 快。

⇒ 所以默认 V=4096、稠密梯度。
"""
from __future__ import annotations
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
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_spatial_key(tokens, grid, table_size, mode='4gram'):
    """由 token 矩阵算静态路由键（纯整数运算，无参数）。

    tokens: (B, T) int64，T = grid*grid，光栅顺序
    返回:   (B, T) int64，取值 [0, table_size)

    mode:
      '1gram'  只用当前 token（键空间 = 词表大小）
      '2gram'  当前 + 左邻
      '4gram'  当前 + 左 + 上 + 上右
      '5gram'  再加 上左
    越界用 0 填充（等价 BOS）。
    """
    B, T = tokens.shape
    g = grid
    x = tokens.reshape(B, g, g).to(torch.int64)
    pad = torch.zeros(B, g + 1, g + 2, dtype=torch.int64, device=x.device)
    pad[:, 1:, 1:g + 1] = x
    p = pad
    a = p[:, 0:g, 0:g]        # 上左
    b = p[:, 0:g, 1:g + 1]    # 上
    c = p[:, 0:g, 2:g + 2]    # 上右
    d = p[:, 1:g + 1, 0:g]    # 左
    e = x                     # 当前

    if mode == '1gram':
        h = e
    elif mode == '2gram':
        h = (e * 1000003 + d) % 2147483647
    elif mode == '5gram':
        h = (e * 1000003 + d * 1000033 + b * 1000037 + c * 1000039 + a * 1000081) % 2147483647
    else:  # 4gram
        h = (e * 1000003 + d * 1000033 + b * 1000037 + c * 1000039) % 2147483647
    return (h % table_size).reshape(B, T)


class L3FFN(nn.Module):
    """L³ 层，替代 SwiGLU。键由调用方算好传入（不依赖 hidden state）。"""

    def __init__(self, d_model, table_size=4096, k_emb=8, d_emb=128, d_up=128,
                 sparse=False):
        super().__init__()
        self.d_model = d_model
        self.table_size = table_size
        self.k = k_emb
        self.sparse = sparse
        self.WK = nn.Embedding(table_size, k_emb * d_model, sparse=sparse)
        self.WV = nn.Embedding(table_size, k_emb * d_emb, sparse=sparse)
        self.W_up = nn.Linear(d_emb, d_up)
        self.norm = nn.LayerNorm(d_up)
        self.W_mix = nn.Linear(d_up + d_model, d_model)
        nn.init.normal_(self.WK.weight, std=0.02)
        nn.init.normal_(self.WV.weight, std=0.02)

    def forward(self, x, keys):
        """
        x:    (B, T, d_model)
        keys: (B, T) int64 静态路由键
        """
        B, T, D = x.shape
        K = self.WK(keys).view(B, T, self.k, D)
        V = self.WV(keys).view(B, T, self.k, -1)
        s = torch.einsum('btkd,btd->btk', K, x)
        a = torch.softmax(s, dim=-1)
        z = torch.einsum('btk,btke->bte', a, V)
        u = self.norm(self.W_up(z))
        return self.W_mix(torch.cat([u, x], dim=-1))

    def sparse_params(self):
        """返回应该用 SparseAdam 优化的参数（sparse=True 时必须分开优化）。"""
        return list(self.WK.parameters()) + list(self.WV.parameters())

    def dense_params(self):
        out = []
        for n, p in self.named_parameters():
            if not n.startswith(('WK', 'WV')):
                out.append(p)
        return out


def build_keys(tokens, grid, table_size, mode, use_bos=False):
    """算静态键。use_bos 时序列前面多一个位置，键也补一个（用 key 0）。"""
    k = causal_spatial_key(tokens, grid, table_size, mode)
    if use_bos:
        B = k.shape[0]
        bos_key = torch.zeros(B, 1, dtype=k.dtype, device=k.device)
        k = torch.cat([bos_key, k[:, :-1]], dim=1)
    return k


if __name__ == '__main__':
    # 自检
    torch.manual_seed(0)
    B, G = 4, 16
    T = G * G
    x = torch.randn(B, T, 512)
    tok = torch.randint(0, 1024, (B, T))
    m = L3FFN(512, table_size=4096)
    for mode in ('1gram', '4gram'):
        k = build_keys(tok, G, 4096, mode)
        y = m(x, k)
        print('  %-7s keys %s -> out %s  (表 %.1fM 参数)'
              % (mode, tuple(k.shape), tuple(y.shape),
                 sum(p.numel() for p in m.parameters()) / 1e6))
    # use_bos 版本
    k = build_keys(tok, G, 4096, '4gram', use_bos=True)
    print('  use_bos keys %s' % (tuple(k.shape),))
    print('  自检通过')
