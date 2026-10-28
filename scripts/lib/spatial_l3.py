# -*- coding: utf-8 -*-
"""spatial_l3.py — 空间 L³ 层原型

论文公式（arXiv:2601.21461）：
    L³(x, t) = W_mix · [ LayerNorm(W_up(V_t^T · Softmax(K_t · x))) ; x ]

与文本版的关键区别 —— **静态路由键用「因果空间邻域」而不是 LZW 后缀**：

光栅顺序下，位置 (r,c) 的因果邻域（生成时已知）：
        (r-1,c-1)  (r-1,c)  (r-1,c+1)
        (r,c-1)    [r,c] 本身也是已知输入
所以 key 可以由这 5 个 token 的局部模式算出，**不需要 hidden state**，
于是参数可以预取（L³ 的核心系统优势）。

我们的 VQ 词表只有 1024，单 token 键空间太小；用空间 n-gram 哈希
把键空间撑到 V（可调），稀疏率随 V 线性增长。
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


def causal_spatial_key(tokens, grid, vocab, table_size, mode='4gram'):
    """由 token 矩阵算静态路由键（纯整数运算，无参数，CPU 友好）。

    tokens: (B, T) int64，T = grid*grid，光栅顺序
    返回:   (B, T) int64，取值 [0, table_size)

    邻域（光栅序下均已生成）：
        a = 上左 (r-1, c-1)
        b = 上   (r-1, c)
        c = 上右 (r-1, c+1)
        d = 左   (r, c-1)
        e = 当前 (r, c)      ← 输入 token，AR 里已知
    越界位置用 0 填充（等价于 BOS）。
    """
    B, T = tokens.shape
    g = grid
    x = tokens.reshape(B, g, g).to(torch.int64)
    pad = torch.zeros(B, g + 1, g + 2, dtype=torch.int64, device=x.device)
    pad[:, 1:, 1:g + 1] = x                      # 放在右下，留出上/左边界
    # 现在 x[r,c] 对应 pad[r+1, c+1]
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
    elif mode == '4gram':
        h = (e * 1000003 + d * 1000033 + b * 1000037 + c * 1000039) % 2147483647
    else:  # 5gram
        h = (e * 1000003 + d * 1000033 + b * 1000037 + c * 1000039 + a * 1000081) % 2147483647
    return (h % table_size).reshape(B, T)


class SpatialL3(nn.Module):
    """用空间 n-gram 静态路由的 Large Lookup Layer。"""

    def __init__(self, d_model, grid, vocab=1024, table_size=65536,
                 k_emb=8, d_emb=128, d_up=128, key_mode='4gram'):
        super().__init__()
        self.d_model = d_model
        self.grid = grid
        self.vocab = vocab
        self.table_size = table_size
        self.k_emb = k_emb
        self.d_emb = d_emb
        self.d_up = d_up
        self.key_mode = key_mode

        # 每张表：table_size 行，每行 k_emb 个向量
        self.WK = nn.Embedding(table_size, k_emb * d_model)
        self.WV = nn.Embedding(table_size, k_emb * d_emb)
        self.W_up = nn.Linear(d_emb, d_up)
        self.norm = nn.LayerNorm(d_up)
        self.W_mix = nn.Linear(d_up + d_model, d_model)

        nn.init.normal_(self.WK.weight, std=0.02)
        nn.init.normal_(self.WV.weight, std=0.02)

    def forward(self, x, tokens):
        """
        x:      (B, T, d_model)  hidden state，T = grid*grid
        tokens: (B, T) int64     该位置的输入 token（AR 中已知）
        """
        B, T, D = x.shape
        key = causal_spatial_key(tokens, self.grid, self.vocab,
                                 self.table_size, self.key_mode)     # (B, T)
        K = self.WK(key).view(B, T, self.k_emb, D)          # (B,T,k,D)
        V = self.WV(key).view(B, T, self.k_emb, self.d_emb)  # (B,T,k,d_emb)

        # 用 hidden state 对这些 embedding 做 attention（context-dependent 聚合）
        scores = torch.einsum('btkd,btd->btk', K, x)         # (B,T,k)
        attn = torch.softmax(scores, dim=-1)
        z = torch.einsum('btk,btke->bte', attn, V)           # (B,T,d_emb)

        u = self.norm(self.W_up(z))
        return self.W_mix(torch.cat([u, x], dim=-1))


if __name__ == '__main__':
    import time
    torch.set_num_threads(6)

    B, G, D = 8, 16, 512
    T = G * G
    x = torch.randn(B, T, D)
    tok = torch.randint(0, 1024, (B, T))

    print('=' * 68)
    print('空间 L³ 原型 —— 键空间与稀疏性检查')
    print('=' * 68)
    for mode in ('1gram', '2gram', '4gram', '5gram'):
        for V in (4096, 65536, 1048576):
            k = causal_spatial_key(tok, G, 1024, V, mode)
            uniq = int(k.unique().numel())
            print('  %-7s V=%-8d 本 batch 命中 %6d 个键 (%.1f%%)'
                  % (mode, V, uniq, 100 * uniq / V))
    print()
    print('  说明：命中率低 = 稀疏性好 = 活跃参数少')
