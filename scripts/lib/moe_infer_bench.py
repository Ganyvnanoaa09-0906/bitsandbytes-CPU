# -*- coding: utf-8 -*-
"""moe_infer_bench.py — inference-side 3-bit MoE experts: fp32 vs materialize vs fused GEMM.

KTransformers-style CPU MoE inference keeps experts quantized and offloaded; to compare on a
reproducible basis we first need to know what a *local* 3-bit expert path costs on this CPU.
This benchmark uses the scaled Mixtral (HF transformers 5.16 fused 3D tensor experts) in eval
mode and compares three expert implementations, all with the same router:

  A. fp32      : original MixtralExperts forward (F.linear on fp32 weights)
  B. materialize: experts stored 3-bit; per hit expert dequantize the whole matrix with the
                  AVX2 dequant kernel into fp32, then F.linear (the "old" 3-bit path)
  C. fused     : experts stored 3-bit; per hit expert call q3bit_gemm directly on the packed
                  matrix (A @ dequant(B)^T inside the kernel; no fp32 weight materialization)

Reports prefill (seq=64) and decode (seq=1) forward time, tokens/s, expert weight memory and
RSS. This is the honest, reproducible baseline for any KTransformers inference comparison.
"""
from __future__ import annotations
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

import ctypes
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from q3bit_proto import load_kernel, quantize_blocks  # noqa: E402

torch.set_num_threads(6)

GEMM_LIB = ctypes.CDLL(r'D:\work\bitsandbytes-CPU\q3bit_gemm.dll')
GEMM_LIB.q3bit_gemm.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
GEMM_LIB.q3bit_gemm.restype = ctypes.c_int
DEQ_LIB = load_kernel()

BLOCK = 64


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1 << 30)
    except Exception:
        return -1.0


def weight_mem_gb(model):
    tot = sum(p.numel() * p.element_size() for p in model.parameters())
    tot += sum(b.numel() * b.element_size() for b in model.buffers())
    return tot / (1 << 30)


def build_scaled_moe():
    from transformers import MixtralConfig, MixtralForCausalLM
    cfg = MixtralConfig(vocab_size=2000, hidden_size=512, intermediate_size=1024,
                        num_hidden_layers=4, num_attention_heads=8,
                        num_key_value_heads=2, num_local_experts=8,
                        num_experts_per_tok=2, max_position_embeddings=128)
    torch.manual_seed(0)
    m = MixtralForCausalLM(cfg)
    m.eval()
    return m


class QuantExperts(nn.Module):
    """Drop-in MixtralExperts replacement; mode='materialize' or 'fused'."""

    def __init__(self, orig, mode='fused', block=BLOCK, nthreads=1, ntile=32):
        super().__init__()
        self.num_experts = orig.num_experts
        self.hidden_dim = orig.hidden_dim
        self.intermediate_dim = orig.intermediate_dim
        self.mode = mode
        self.block = block
        self.nthreads = nthreads
        self.ntile = ntile
        self.gu_shape = (2 * self.intermediate_dim, self.hidden_dim)
        self.down_shape = (self.hidden_dim, self.intermediate_dim)
        for e in range(self.num_experts):
            for tag, mat in (('gu', orig.gate_up_proj[e].detach()),
                             ('down', orig.down_proj[e].detach())):
                packed, absmax, shape, n = quantize_blocks(mat, 3, block)
                self.register_buffer(f'{tag}_packed_{e}', packed)
                self.register_buffer(f'{tag}_absmax_{e}', absmax)

    def _deq(self, e, tag, shape):
        packed = getattr(self, f'{tag}_packed_{e}')
        absmax = getattr(self, f'{tag}_absmax_{e}')
        out = torch.empty(shape[0] * shape[1], dtype=torch.float32)
        DEQ_LIB.bnb_dequant_3bit(packed.data_ptr(), absmax.data_ptr(), out.data_ptr(),
                                 absmax.numel(), self.block)
        return out.reshape(shape)

    def _fused_gemm(self, x, e, tag, shape):
        x = x.contiguous()
        out = torch.empty(x.shape[0], shape[0], dtype=torch.float32)
        packed = getattr(self, f'{tag}_packed_{e}')
        absmax = getattr(self, f'{tag}_absmax_{e}')
        rc = GEMM_LIB.q3bit_gemm(x.data_ptr(), x.shape[0], packed.data_ptr(),
                                 absmax.data_ptr(), out.data_ptr(), shape[0],
                                 shape[1], self.block, self.nthreads, self.ntile)
        assert rc == 0
        return out

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            if self.mode == 'materialize':
                gu = self._deq(e, 'gu', self.gu_shape)
                gate, up = F.linear(current_state, gu).chunk(2, dim=-1)
                current = F.linear(F.silu(gate) * up, self._deq(e, 'down', self.down_shape))
            else:
                gu = self._fused_gemm(current_state, e, 'gu', self.gu_shape)
                gate, up = gu.chunk(2, dim=-1)
                current = self._fused_gemm(F.silu(gate) * up, e, 'down', self.down_shape)
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current)
        return final_hidden_states


def replace_experts(model, mode: str, nthreads=1):
    n = 0
    for layer in model.model.layers:
        layer.mlp.experts = QuantExperts(layer.mlp.experts, mode=mode, nthreads=nthreads)
        n += 1
    return n


def bench_forward(model, seq: int, iters: int = 5):
    ids = torch.randint(1, model.config.vocab_size, (1, seq))
    with torch.no_grad():
        for _ in range(2):
            model(input_ids=ids)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(input_ids=ids)
        dt = (time.perf_counter() - t0) / iters
    return dt * 1e3


def run_variant(name, mode=None, nthreads=1):
    model = build_scaled_moe()
    if mode is not None:
        replace_experts(model, mode, nthreads=nthreads)
    mem = weight_mem_gb(model)
    prefill = bench_forward(model, 64)
    decode = bench_forward(model, 1)
    print('%-26s weight-mem %.3f GB | prefill(64) %7.1f ms (%6.1f tok/s) | decode(1) %6.2f ms | rss %.2f GB'
          % (name, mem, prefill, 64 / (prefill / 1e3), decode, rss_gb()))
    return {'name': name, 'mem': mem, 'prefill_ms': prefill, 'decode_ms': decode}


def main():
    results = []
    results.append(run_variant('fp32 experts'))
    results.append(run_variant('3-bit materialize', mode='materialize'))
    results.append(run_variant('3-bit fused GEMM (1 thread)', mode='fused', nthreads=1))
    results.append(run_variant('3-bit fused GEMM (4 threads)', mode='fused', nthreads=4))
    print()
    for r in results:
        print('RESULT %-26s prefill_ms=%.1f decode_ms=%.2f mem_gb=%.3f'
              % (r['name'], r['prefill_ms'], r['decode_ms'], r['mem']))
    print('MOE INFER DONE')


if __name__ == '__main__':
    main()
