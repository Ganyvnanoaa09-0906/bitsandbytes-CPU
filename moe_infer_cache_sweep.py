# -*- coding: utf-8 -*-
"""moe_infer_cache_sweep.py — memory-budget vs throughput curve for CPU MoE inference.

KTransformers-style offload answers "how many experts can I keep hot under a memory budget".
This script produces the resident quantization analogue on the scaled Mixtral (53.5M):

  * experts are stored 3-bit (packed, ~11% of fp32 expert memory);
  * the top-C experts per layer (by router usage) are dequantized once and cached fp32;
  * the remaining (8-C) experts are dequantized per call (materialize + oneDNN matmul).

Sweeping C = 0,2,4,6,8 gives a real memory-vs-speed curve: weight memory vs prefill/decode
latency and tokens/s, with the same weights and router.
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from q3bit_proto import load_kernel, quantize_blocks  # noqa: E402
from moe_infer_bench import (BLOCK, DEQ_LIB, build_scaled_moe,  # noqa: E402
                             bench_forward, rss_gb, weight_mem_gb)

torch.set_num_threads(6)

FP32_MEM = 0.205  # measured fp32 expert model weight memory (GB)


def router_usage(model, batches):
    from collections import Counter
    counts = {}

    def hook(mod, inp, out):
        idx = out[2] if isinstance(out, (tuple, list)) and len(out) > 2 else None
        if idx is None:
            return
        cnt = counts.setdefault(hook.name, Counter())
        for v in idx.reshape(-1).tolist():
            cnt[int(v)] += 1

    handles = []
    for name, mod in model.named_modules():
        if type(mod).__name__.endswith('Router'):
            hook.name = name
            handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for b in batches:
            model(**b)
    for h in handles:
        h.remove()
    return {k: [e for e, _ in v.most_common()] for k, v in counts.items()}


class CacheExperts(nn.Module):
    """3-bit experts + top-C hot experts cached in fp32 (per layer)."""

    def __init__(self, orig, hot_count: int, hot_order, block=BLOCK):
        super().__init__()
        self.num_experts = orig.num_experts
        self.hidden_dim = orig.hidden_dim
        self.intermediate_dim = orig.intermediate_dim
        self.block = block
        self.hot = set(int(e) for e in hot_order[:hot_count])
        self.gu_shape = (2 * self.intermediate_dim, self.hidden_dim)
        self.down_shape = (self.hidden_dim, self.intermediate_dim)
        for e in range(self.num_experts):
            for tag, mat in (('gu', orig.gate_up_proj[e].detach()),
                             ('down', orig.down_proj[e].detach())):
                packed, absmax, shape, n = quantize_blocks(mat, 3, block)
                if e in self.hot:
                    # cache the dequantized fp32 matrix (hot expert)
                    w = torch.empty(shape[0] * shape[1], dtype=torch.float32)
                    DEQ_LIB.bnb_dequant_3bit(packed.data_ptr(), absmax.data_ptr(),
                                             w.data_ptr(), absmax.numel(), block)
                    self.register_buffer(f'{tag}_hot_{e}', w.reshape(shape))
                else:
                    self.register_buffer(f'{tag}_packed_{e}', packed)
                    self.register_buffer(f'{tag}_absmax_{e}', absmax)

    def _weights(self, e, tag, shape):
        if e in self.hot:
            return getattr(self, f'{tag}_hot_{e}')
        packed = getattr(self, f'{tag}_packed_{e}')
        absmax = getattr(self, f'{tag}_absmax_{e}')
        out = torch.empty(shape[0] * shape[1], dtype=torch.float32)
        DEQ_LIB.bnb_dequant_3bit(packed.data_ptr(), absmax.data_ptr(), out.data_ptr(),
                                 absmax.numel(), self.block)
        return out.reshape(shape)

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
            gu = self._weights(e, 'gu', self.gu_shape)
            gate, up = F.linear(current_state, gu).chunk(2, dim=-1)
            down = self._weights(e, 'down', self.down_shape)
            current = F.linear(F.silu(gate) * up, down)
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current)
        return final_hidden_states


def run(cache_per_layer: int):
    model = build_scaled_moe()
    calib = [{'input_ids': torch.randint(1, model.config.vocab_size, (1, 32))}
             for _ in range(4)]
    order = router_usage(model, calib)
    for layer in model.model.layers:
        # match the router module by object identity to its usage-order entry
        hot_order = list(range(model.config.num_local_experts))
        for n, m in model.named_modules():
            if m is layer.mlp.gate and n in order:
                hot_order = order[n]
                break
        layer.mlp.experts = CacheExperts(layer.mlp.experts, cache_per_layer, hot_order)
    mem = weight_mem_gb(model)
    prefill = bench_forward(model, 64)
    decode = bench_forward(model, 1)
    saved = 100.0 * (1 - mem / FP32_MEM)
    print('C=%-2d hot/layer | weight-mem %.3f GB (%.0f%% of fp32) | prefill %6.1f ms (%5.0f tok/s) | decode %5.2f ms | rss %.2f GB'
          % (cache_per_layer, mem, 100 * mem / FP32_MEM, prefill, 64 / (prefill / 1e3),
             decode, rss_gb()))
    return cache_per_layer, mem, prefill, decode


def main():
    print('=== memory-budget sweep: hot fp32 experts + cold 3-bit (scaled Mixtral) ===')
    print('fp32 baseline: %.3f GB, prefill 43.1 ms, decode 11.96 ms' % FP32_MEM)
    for c in (0, 2, 4, 6, 8):
        run(c)
    print('MOE CACHE SWEEP DONE')


if __name__ == '__main__':
    main()
