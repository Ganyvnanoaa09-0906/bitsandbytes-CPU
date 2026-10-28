# -*- coding: utf-8 -*-
"""make_coarse_cond.py — 把 Stage-1 的 16x16 粗 token 变成 Stage-2 的 cross-attn 条件。

两阶段 AR 的思路（GPT 的建议 + 我们自己的测量）：
  Stage 1 已经会**构图**（§10.17 证明 T=256 flat 的构图是好的）；
  Stage 2 只需**补细节**，于是给它一个 T=1024 的序列，但把粗布局作为条件喂进去，
  它就不必再自己"想清楚人在哪里"。

条件形式：粗 token 的【码本嵌入】序列 (N, 256, 64)
  —— 用 C_mse 码本查表，维度 64 = 该分词器的 embed_dim，
     配合 --cond-mode cross --text-dim 64 使用（cross 会用 cond_proj 把它投到 d）。

为什么在容器上算而不是上传：fp16 也有 954 MB，上传太慢；
而容器上已经有 tokens_16x16_full.pt 与 vqvae_16x16_cmse.pt。
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
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True, help='Stage-1 的 (N,256) 粗 token 文件')
    ap.add_argument('--vqvae', required=True, help='Stage-1 用的 VQVAE（含码本）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 行（冒烟用）')
    a = ap.parse_args()

    ck = torch.load(a.vqvae, map_location='cpu', weights_only=False)
    vq, _ = vqvae_from_ckpt(ck)
    W = vq.vq.codebook.weight.data.float()          # (n_codes, embed_dim)
    print('[coarse] codebook %s' % (tuple(W.shape),))

    t = torch.load(a.tokens, map_location='cpu', weights_only=False).long()
    if a.limit:
        t = t[:a.limit]
    print('[coarse] tokens %s  min %d max %d' % (tuple(t.shape), int(t.min()), int(t.max())))
    assert int(t.max()) < W.shape[0], '粗 token 超出码本范围'

    emb = W[t]                                       # (N, 256, embed_dim)
    print('[coarse] cond %s  mean %.4f std %.4f' % (tuple(emb.shape), float(emb.mean()), float(emb.std())))
    torch.save(emb.half(), a.out)
    print('[coarse] -> %s  %.1f MB' % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == '__main__':
    main()
