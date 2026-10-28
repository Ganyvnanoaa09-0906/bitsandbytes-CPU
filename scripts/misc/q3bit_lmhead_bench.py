# -*- coding: utf-8 -*-
"""q3bit_lmhead_bench.py — is a 3-bit tied lm_head worth it?

Qwen3.5-0.8B ties lm_head with the text embedding table (248320 x 1024 fp32 = ~1 GB).
If we quantize the embedding, the final logits matmul needs the full table dequantized for
every step. This script measures the real cost on CPU with the csrc-style AVX2 kernel:
  * fp32 lm_head: matmul only
  * 3-bit lm_head: kernel dequant of the full table + the same matmul
and reports memory saved vs per-step time overhead, so the decision is data-driven.

Usage: python q3bit_lmhead_bench.py
"""
from __future__ import annotations
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

import ctypes
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.q3bit_proto import QuantEmbedding, load_kernel


def bench_vocab(vocab: int, dim: int, seq: int = 64, block: int = 64):
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        r'D:\deepsleep\_test\models\qwen3.5-0.8B', dtype=torch.float32,
        low_cpu_mem_usage=True)
    emb = model.model.language_model.embed_tokens
    w = emb.weight.detach().clone()  # (vocab, dim) fp32
    print('lm_head table: %s fp32 = %.3f GB' % (tuple(w.shape), w.numel() * 4 / (1 << 30)))

    hidden = torch.randn(seq, dim) * 0.1
    hb = hidden.to(torch.float32)

    def fp32_logits():
        return F.linear(hb, w)

    # warmup
    for _ in range(2):
        fp32_logits()
    t0 = time.perf_counter()
    for _ in range(5):
        fp32_logits()
    t_fp32 = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    qemb = QuantEmbedding(emb, bits=3, block=block)
    t_quant = time.perf_counter() - t0
    packed_bytes = qemb.packed.numel() + qemb.absmax.numel() * 4
    print('3-bit quantize: %.1f s | packed=%.3f GB (%.1f%% of fp32)'
          % (t_quant, packed_bytes / (1 << 30), 100.0 * packed_bytes / (w.numel() * 4)))

    lib = load_kernel()
    assert lib is not None, 'q3bit_kernel dll missing'
    nb_h = qemb.padded_dim // block
    n_blocks = qemb.num_embeddings * nb_h
    out = torch.empty(qemb.num_embeddings * qemb.padded_dim, dtype=torch.float32)

    def dequant_full():
        lib.bnb_dequant_3bit(qemb.packed.contiguous().data_ptr(),
                             qemb.absmax.contiguous().data_ptr(), out.data_ptr(),
                             n_blocks, block)
        return out

    # warmup + time pure dequant
    dequant_full()
    t0 = time.perf_counter()
    for _ in range(3):
        dequant_full()
    t_dequant = (time.perf_counter() - t0) / 3
    deq_w = out.reshape(qemb.num_embeddings, qemb.padded_dim)[:, :dim]

    # correctness of the dequantized table vs original fp32
    err = (deq_w - w).norm() / w.norm()
    print('dequant relL2 = %.4f | dequant time %.1f ms' % (float(err), t_dequant * 1e3))

    def quant_logits():
        dequant_full()
        return F.linear(hb, deq_w)

    quant_logits()  # warmup
    t0 = time.perf_counter()
    for _ in range(5):
        quant_logits()
    t_quant_logits = (time.perf_counter() - t0) / 5

    print()
    print('%-28s %10s %10s' % ('', 'fp32', '3-bit'))
    print('%-28s %10.1f %10.1f' % ('logits time (ms/step)', t_fp32 * 1e3, t_quant_logits * 1e3))
    print('%-28s %10.3f %10.3f' % ('lm_head memory GB', w.numel() * 4 / (1 << 30),
                                   packed_bytes / (1 << 30)))
    print('%-28s %10s %10.1f' % ('dequant time (ms)', '-', t_dequant * 1e3))
    print('%-28s %10.1f %10.1f' % ('overhead vs fp32 (ms/step)', 0.0,
                                   (t_quant_logits - t_fp32) * 1e3))
    print('%-28s %10s %9.1f%%' % ('memory saving', '-',
                                  100.0 * (1 - packed_bytes / (w.numel() * 4))))
    print('RESULT lm_head model=qwen0.8b seq=%d fp32_ms=%.1f quant_ms=%.1f mem_fp32_gb=%.3f '
          'mem_3bit_gb=%.3f' % (seq, t_fp32 * 1e3, t_quant_logits * 1e3,
                                w.numel() * 4 / (1 << 30), packed_bytes / (1 << 30)))


def main():
    torch.set_num_threads(6)
    bench_vocab(248320, 1024, seq=64)


if __name__ == '__main__':
    main()
