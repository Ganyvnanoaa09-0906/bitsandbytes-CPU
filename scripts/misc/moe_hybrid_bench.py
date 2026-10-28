# -*- coding: utf-8 -*-
"""moe_hybrid_bench.py — EFST hot-expert selection + cold-expert 3-bit quantization.

Real HF transformers 5.16 Mixtral keeps experts as fused 3D tensors:
  MixtralExperts.gate_up_proj [E, 2I, H], MixtralExperts.down_proj [E, H, I],
  with a custom MixtralTopKRouter. EFST supports exactly this tensor-expert layout, so this
experiment:
  1. builds a scaled synthetic Mixtral (hidden 512 / 4 layers / 8 experts / top-2, ~53M params);
  2. applies EFST (top_k=2, no LoRA) on calibration batches to pick the hottest experts;
  3. replaces each MixtralExperts with HybridExperts: hot experts stay fp32, cold experts are
     stored 3-bit and dequantized on demand (AVX2 kernel) during the expert dispatch loop;
  4. trains frozen base + attention LoRA for a few steps and compares vs the fp32 baseline and
     the all-cold (all 3-bit) variant: weight memory, RSS, step time, CE loss, routing.

A null/negative result is a valid outcome: if all experts are hit every step, cold dequant
costs the same as full 3-bit and hybrid only wins on memory.
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

import os
import sys
import time

import torch
import torch.nn
import torch.nn.functional

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

import bitsandbytes
from bitsandbytes.efst import EFSTConfig, apply_efst  # noqa: E402
from scripts.lib.q3bit_proto import load_kernel, quantize_blocks

torch.set_num_threads(6)


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1 << 30)
    except Exception:
        return -1.0


def model_mem_gb(model):
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
    return MixtralForCausalLM(cfg)


def count_router_usage(model, batches):
    """Average tokens routed to each expert per layer, from real router calls."""
    from collections import Counter
    counts = {}

    def make_hook(name):
        def hook(mod, inp, out):
            idx = None
            if isinstance(out, (tuple, list)):
                for item in out:
                    if isinstance(item, torch.Tensor) and item.dim() == 2 and \
                            item.dtype in (torch.int64, torch.int32, torch.long):
                        idx = item
                        break
            if idx is None and isinstance(out, torch.Tensor) and out.dim() == 2:
                idx = torch.topk(out, 2, dim=-1).indices
            if idx is not None:
                cnt = counts.setdefault(name, Counter())
                for v in idx.reshape(-1).tolist():
                    cnt[int(v)] += 1
        return hook

    handles = []
    for name, mod in model.named_modules():
        if type(mod).__name__.endswith('Router'):
            handles.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        for b in batches:
            model(**b)
    for h in handles:
        h.remove()
    return counts


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale


def wrap_attention_lora(model):
    n = 0
    for name, mod in list(model.named_modules()):
        leaf = name.split('.')[-1]
        if isinstance(mod, nn.Linear) and leaf in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            parent = model.get_submodule(name.rpartition('.')[0])
            setattr(parent, leaf, LoRALinear(mod))
            n += 1
    return n


class HybridExperts(nn.Module):
    """MixtralExperts replacement: hot experts fp32, cold experts 3-bit (on-demand dequant)."""

    def __init__(self, orig, hot, bits=3, block=64):
        super().__init__()
        self.num_experts = orig.num_experts
        self.hidden_dim = orig.hidden_dim
        self.intermediate_dim = orig.intermediate_dim
        self.bits = bits
        self.block = block
        self.hot = set(int(h) for h in hot)
        for e in range(self.num_experts):
            if e in self.hot:
                setattr(self, f'gu_{e}', nn.Parameter(
                    orig.gate_up_proj[e].detach().clone(), requires_grad=False))
                setattr(self, f'down_{e}', nn.Parameter(
                    orig.down_proj[e].detach().clone(), requires_grad=False))
            else:
                for tag, mat in (('gu', orig.gate_up_proj[e].detach()),
                                 ('down', orig.down_proj[e].detach())):
                    packed, absmax, shape, n = quantize_blocks(mat, bits, block)
                    self.register_buffer(f'{tag}_packed_{e}', packed)
                    self.register_buffer(f'{tag}_absmax_{e}', absmax)
                    self.register_buffer(f'{tag}_shape_{e}', torch.tensor(shape, dtype=torch.long))

    def _weights(self, e):
        if e in self.hot:
            return getattr(self, f'gu_{e}'), getattr(self, f'down_{e}')
        lib = load_kernel()
        out = {}
        for tag, out_shape in (('gu', (2 * self.intermediate_dim, self.hidden_dim)),
                               ('down', (self.hidden_dim, self.intermediate_dim))):
            packed = getattr(self, f'{tag}_packed_{e}')
            absmax = getattr(self, f'{tag}_absmax_{e}')
            w = torch.empty(out_shape[0] * out_shape[1], dtype=torch.float32)
            nblocks = absmax.numel()
            fn = lib.bnb_dequant_3bit if self.bits == 3 else lib.bnb_dequant_2bit
            fn(packed.data_ptr(), absmax.data_ptr(), w.data_ptr(), nblocks, self.block)
            out[tag] = w.reshape(out_shape)
        return out['gu'], out['down']

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gu, down = self._weights(expert_idx)
            gate, up = F.linear(current_state, gu).chunk(2, dim=-1)
            current = F.linear(F.silu(gate) * up, down)
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states


def build_variant(variant: str, steps: int = 3):
    from transformers import MixtralForCausalLM
    model = build_scaled_moe()
    model.train()
    ids = torch.randint(1, model.config.vocab_size, (1, 64))
    labels = ids.clone()
    calib = [{'input_ids': torch.randint(1, model.config.vocab_size, (1, 32))} for _ in range(4)]

    n_experts = model.config.num_local_experts
    if variant == 'fp32':
        for p in model.parameters():
            p.requires_grad_(False)
        wrap_attention_lora(model)
    else:
        assert variant in ('all3', 'hybrid'), variant
        res = apply_efst(model, EFSTConfig(
            top_k=2, calibration_dataloader=calib,
            calibration_forward_fn=lambda b: model(**b),
            tune_router=False, lora=False, allow_no_experts=False))
        hot = {}
        for path, keys in res.selected_experts.items():
            idx = []
            for k in keys:
                try:
                    idx.append(int(k))
                except (TypeError, ValueError):
                    pass
            hot[path] = idx
        print('  EFST selected:', {k.split('layers.')[-1]: v for k, v in hot.items()})
        for i, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            path = None
            for p, keys in hot.items():
                if f'layers.{i}.' in p:
                    path = keys
                    break
            chosen = path if path is not None else []
            if variant == 'all3':
                chosen = []
            mlp.experts = HybridExperts(mlp.experts, chosen)
        # drop EFST's references to the original fp32 expert tensors before measuring RSS
        del res
        import gc
        gc.collect()
        # freeze everything, then LoRA on attention
        for p in model.parameters():
            p.requires_grad_(False)
        wrap_attention_lora(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    usage = count_router_usage(model, [{'input_ids': ids}])
    init_mem = model_mem_gb(model)
    opt = bnb.optim.AdamW8bit(trainable, lr=1e-4)
    times, losses = [], []
    for step in range(steps + 1):
        t0 = time.perf_counter()
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        if step:
            times.append(dt)
            losses.append(float(loss.detach()))
    med = sorted(times)[len(times) // 2] * 1e3
    print('=== %s: weight-mem %.3f GB | step %.1f ms | rss %.2f GB | loss %.4f -> %.4f'
          % (variant, init_mem, med, rss_gb(), losses[0], losses[-1]))
    print('  router token counts (first layers):',
          {k.split('layers.')[-1]: dict(sorted(v.items())) for k, v in list(usage.items())[:2]})
    print('RESULT %s step_ms=%.1f weight_mem=%.3f rss=%.2f loss_first=%.4f loss_last=%.4f'
          % (variant, med, init_mem, rss_gb(), losses[0], losses[-1]))
    return model


def main():
    build_variant('fp32', steps=3)
    print()
    build_variant('all3', steps=3)
    print()
    build_variant('hybrid', steps=3)
    print('\nMOE HYBRID DONE')


if __name__ == '__main__':
    main()
