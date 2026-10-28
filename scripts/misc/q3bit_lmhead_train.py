# -*- coding: utf-8 -*-
"""q3bit_lmhead_train.py — end-to-end lm_head comparison for 3-bit CPU training.

Variants (same Qwen3.5-0.8B, seq=64, LoRA, 2 steps, AdamW8bit):
  A. fp32 frozen base + fp32 lm_head (normal torch path)
  B. 3-bit linear+embedding + fused 3-bit lm_head (forward q3bit_gemm, backward q3bit_gemm_t)

Reports per-step time, weight memory, RSS and the CE loss trajectory. The goal is the honest
full-step tradeoff, since the lm_head backward GEMM (dX = dY @ W) is compute-bound and our
fused transposed kernel is slower than torch's oneDNN matmul.
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import argparse
import ctypes
import gc
import os
import sys
import time

import torch
import torch.nn
import torch.nn.functional

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
from scripts.lib.q3bit_proto import QuantEmbedding, QuantLinear, quantize_embeddings, quantize_model, model_mem_gb, rss_gb, load_kernel

torch.set_num_threads(6)

_LIB = ctypes.CDLL(r'D:\work\bitsandbytes-CPU\q3bit_gemm.dll')
_LIB.q3bit_gemm.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int]
_LIB.q3bit_gemm.restype = ctypes.c_int
_LIB.q3bit_gemm_t.argtypes = _LIB.q3bit_gemm.argtypes
_LIB.q3bit_gemm_t.restype = ctypes.c_int


class _Q3HeadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, packed, absmax):
        M, K = hidden.shape
        N = packed.shape[0]
        hidden = hidden.contiguous()
        out = torch.empty(M, N, dtype=torch.float32)
        rc = _LIB.q3bit_gemm(hidden.data_ptr(), M, packed.data_ptr(), absmax.data_ptr(),
                             out.data_ptr(), N, K, 64, 6, 64)
        assert rc == 0
        ctx.save_for_backward(packed, absmax)
        ctx.K = K
        ctx.N = N
        return out

    @staticmethod
    def backward(ctx, grad_out):
        packed, absmax = ctx.saved_tensors
        N = packed.shape[0]
        grad_out = grad_out.contiguous()
        M = grad_out.shape[0]
        K = ctx.K
        N = ctx.N
        dX = torch.empty(M, K, dtype=torch.float32)
        rc = _LIB.q3bit_gemm_t(grad_out.data_ptr(), M, packed.data_ptr(), absmax.data_ptr(),
                               dX.data_ptr(), N, K, 64, 6, 64)
        assert rc == 0
        return dX, None, None


class QuantLMHead(nn.Module):
    """Shares the quantized embedding table; forward/backward run fused 3-bit GEMMs."""

    def __init__(self, embed: QuantEmbedding):
        super().__init__()
        self.register_buffer('packed', embed.packed.detach())
        self.register_buffer('absmax', embed.absmax.detach())
        self.vocab = embed.num_embeddings
        self.dim = embed.embedding_dim
        self.nblocks = embed.padded_dim // embed.block

    def forward(self, hidden):
        lead = hidden.shape[:-1]
        h = hidden.reshape(-1, self.dim)
        logits = _Q3HeadFn.apply(h, self.packed, self.absmax)
        return logits.reshape(*lead, self.vocab)


def build(bits: int, lmhead_fused: bool):
    from transformers import Qwen3_5ForConditionalGeneration
    outer = Qwen3_5ForConditionalGeneration.from_pretrained(
        r'D:\deepsleep\_test\models\qwen3.5-0.8B', dtype=torch.float32,
        low_cpu_mem_usage=True)
    outer.train()
    for p in outer.parameters():
        p.requires_grad_(False)
    model = outer.model
    n_emb = 0
    if bits < 32:
        load_kernel()  # must be loaded before quantize_model(use_kernel=True), else torch fallback
        n_emb = quantize_embeddings(model, bits, 64)
    n_lin = quantize_model(model, bits, 64, r=8, keep_fp32=(bits >= 32),
                           use_kernel=(bits < 32))
    gc.collect()
    head = None
    if lmhead_fused:
        emb = model.language_model.embed_tokens
        assert isinstance(emb, QuantEmbedding)
        head = QuantLMHead(emb)
        outer.lm_head = None
    else:
        head = outer.lm_head
    # outer owns model; iterating outer.parameters() yields each parameter exactly once
    trainable = [p for p in outer.parameters() if p.requires_grad]
    return outer, model, head, trainable, n_emb, n_lin


def run(args):
    import bitsandbytes
    torch.manual_seed(0)
    t_load = time.time()
    outer, model, head, trainable, n_emb, n_lin = build(args.bits, args.bits < 32)
    opt = bnb.optim.AdamW8bit(trainable, lr=1e-4)
    ids = torch.randint(1, 1000, (1, args.seq))
    labels = ids.clone()
    print('loaded %.1fs | bits=%d | linears %d, embeddings %d | trainable %.2fM | weight-mem %.3f GB | rss %.2f GB'
          % (time.time() - t_load, args.bits, n_lin, n_emb,
             sum(p.numel() for p in trainable) / 1e6, model_mem_gb(outer), rss_gb()))

    times, losses = [], []
    for step in range(args.steps + 1):
        t0 = time.perf_counter()
        h = model(input_ids=ids).last_hidden_state
        logits = head(h)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        if step:
            times.append(dt)
            losses.append(float(loss.detach()))
        if step == 0 or step == args.steps:
            print('  step %d loss %.4f  rss %.2f GB' % (step, float(loss.detach()), rss_gb()))
    med = sorted(times)[len(times) // 2]
    print('RESULT bits=%d seq=%d step_ms=%.0f loss_first=%.4f loss_last=%.4f weight_mem=%.3f rss=%.2f'
          % (args.bits, args.seq, med * 1e3, losses[0], losses[-1],
             model_mem_gb(outer), rss_gb()))
    print('DONE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bits', type=int, default=3, help='3 = quantized, 32 = fp32 baseline')
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--steps', type=int, default=2)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
