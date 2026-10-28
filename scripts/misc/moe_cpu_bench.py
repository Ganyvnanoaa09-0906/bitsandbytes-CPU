# -*- coding: utf-8 -*-
"""moe_cpu_bench.py — CPU MoE baseline (v2).

A. Real tiny Mixtral (hf-mirror) with EFST: transformers 5.x stores Mixtral experts as fused
   3D tensors (MixtralExperts.gate_up_proj [E,2I,H], down_proj [E,H,I]) and a custom router, so
   EFST's tensor-expert path is the relevant integration. We apply EFST (top-2 expert LoRA),
   measure router usage, train a few steps and check the loss falls.
B. Synthetic Mixtral-style MoE built from plain nn.Linear experts (so our 3-bit QuantLinear
   applies): frozen experts + attention-LoRA, fp32 vs 3-bit weights: step time / weight memory /
   RSS / CE loss.
C. Memory-only: quantize the real HF Mixtral fused expert tensors to 3-bit and report the
   expert-only memory reduction (the forward path of fused experts is not patched here).

Honest scope: KTransformers is an inference-time expert-offload server; no local
apples-to-apples baseline exists, so this file establishes a reproducible CPU MoE training
baseline (memory/throughput/convergence) instead of claiming a head-to-head win.
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
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import bitsandbytes
from bitsandbytes.efst import EFSTConfig, apply_efst  # noqa: E402
from scripts.lib.q3bit_proto import QuantLinear, load_kernel, quantize_blocks, quantize_model

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


def count_expert_usage(model, ids):
    """Hook every MoE router and report distinct experts used per layer, averaged over tokens."""
    stats = {}
    handles = []

    def hook(name, n_exp):
        def fn(mod, inp, out):
            idx = None
            if isinstance(out, (tuple, list)):
                for item in out:
                    if isinstance(item, torch.Tensor) and item.dim() == 2 and \
                            item.dtype in (torch.int64, torch.int32, torch.long):
                        idx = item
                        break
            if idx is None and isinstance(out, torch.Tensor) and out.dim() == 2 and                     out.is_floating_point():
                idx = torch.topk(out, min(2, out.shape[-1]), dim=-1).indices
            if idx is not None:
                stats.setdefault(name, []).append(len(torch.unique(idx)))
        return fn

    for name, mod in model.named_modules():
        # HF fused-expert router or a plain gate Linear
        leaf = name.split('.')[-1]
        is_router = type(mod).__name__.endswith('Router') or (
            (isinstance(mod, nn.Linear) or isinstance(mod, QuantLinear))
            and leaf in ('gate', 'router'))
        if not is_router:
            continue
        n_exp = getattr(mod, 'num_experts', None) or getattr(mod, 'out_features', 8)
        handles.append(mod.register_forward_hook(hook(name, n_exp)))
    with torch.no_grad():
        model(input_ids=ids)
    for h in handles:
        h.remove()
    return {k: round(sum(v) / len(v), 1) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# A. real tiny Mixtral + EFST
# ---------------------------------------------------------------------------
def part_a():
    from transformers import AutoModelForCausalLM
    print('=== A. real tiny Mixtral + EFST(top-2 expert LoRA, fused tensor experts) ===')
    model = AutoModelForCausalLM.from_pretrained(
        'hf-internal-testing/tiny-random-MixtralForCausalLM', dtype=torch.float32)
    model.train()
    total = sum(p.numel() for p in model.parameters())
    ids = torch.randint(1, model.config.vocab_size, (2, 16))
    calib = [{'input_ids': torch.randint(1, model.config.vocab_size, (2, 16))} for _ in range(4)]
    res = apply_efst(model, EFSTConfig(
        top_k=2, calibration_dataloader=calib,
        calibration_forward_fn=lambda b: model(**b),
        tune_router=False, lora=True, lora_r=8, lora_alpha=16,
        allow_no_experts=False))
    print(res.summary())
    usage = count_expert_usage(model, ids)
    print('  router usage (distinct experts per token, avg):', usage)
    labels = ids.clone()
    opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    times, losses = [], []
    for step in range(6):
        t0 = time.perf_counter()
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step:
            times.append(time.perf_counter() - t0)
            losses.append(float(loss.detach()))
    med = sorted(times)[len(times) // 2] * 1e3
    print('  EFST train: %.1f ms/step | loss %.4f -> %.4f | rss %.2f GB'
          % (med, losses[0], losses[-1], rss_gb()))
    print('RESULT tiny_moe params=%d trainable=%d step_ms=%.1f loss_first=%.4f loss_last=%.4f expert_groups=%d'
          % (total, res.trainable_after, med, losses[0], losses[-1], len(res.groups)))
    return res


# ---------------------------------------------------------------------------
# C. memory-only quantization of the real HF fused expert tensors
# ---------------------------------------------------------------------------
def part_c():
    from transformers import AutoModelForCausalLM
    print('=== C. real Mixtral fused-expert tensors: 3-bit memory potential (memory only) ===')
    model = AutoModelForCausalLM.from_pretrained(
        'hf-internal-testing/tiny-random-MixtralForCausalLM', dtype=torch.float32)
    fp32_bytes = q3_bytes = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != 'MixtralExperts':
            continue
        for pname, p in mod.named_parameters():
            w = p.detach().float()
            fp32_bytes += w.numel() * 4
            # per-expert 2D quantize: (E, out, in) -> flatten expert dim into rows
            E = w.shape[0]
            mat = w.reshape(E, -1)
            packed, absmax, shape, n = quantize_blocks(mat, 3, 64)
            q3_bytes += packed.numel() + absmax.numel() * 4
    print('  fused expert tensors: fp32 %.1f MB -> 3-bit %.1f MB (%.1f%% of fp32)'
          % (fp32_bytes / 1e6, q3_bytes / 1e6, 100.0 * q3_bytes / fp32_bytes))
    print('RESULT real_moe_expert_mem fp32_mb=%.1f q3_mb=%.1f ratio=%.3f'
          % (fp32_bytes / 1e6, q3_bytes / 1e6, q3_bytes / fp32_bytes))


# ---------------------------------------------------------------------------
# B. synthetic Mixtral-style MoE (nn.Linear experts) fp32 vs 3-bit
# ---------------------------------------------------------------------------
class ToyExpert(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.w1 = nn.Linear(hidden, inter, bias=False)
        self.w2 = nn.Linear(inter, hidden, bias=False)
        self.w3 = nn.Linear(hidden, inter, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ToyMoEBlock(nn.Module):
    def __init__(self, hidden, inter, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_experts = n_experts
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList([ToyExpert(hidden, inter) for _ in range(n_experts)])

    def forward(self, h):
        shape = h.shape
        h2 = h.reshape(-1, shape[-1])
        logits = self.gate(h2)
        probs = F.softmax(logits.float(), dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv.to(h2.dtype)
        out = torch.zeros_like(h2)
        for e in range(self.num_experts):
            mask = (topi == e)
            if not mask.any():
                continue
            rows = torch.where(mask)[0]
            weights = topv[mask].unsqueeze(-1)
            out.index_add_(0, rows, self.experts[e](h2[rows]) * weights)
        return out.reshape(shape)


class ToyMoEModel(nn.Module):
    def __init__(self, vocab=2000, hidden=512, inter=1024, layers=4, experts=8, top_k=2, seq=128):
        super().__init__()
        self.config = type('C', (), {'vocab_size': vocab})()
        self.embed = nn.Embedding(vocab, hidden)
        self.mix = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.layers = nn.ModuleList([ToyMoEBlock(hidden, inter, experts, top_k)
                                     for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, labels=None):
        h = self.embed(input_ids)
        for mix, moe, norm in zip(self.mix, self.layers, self.norms):
            h = norm(h + mix(h))
            h = norm(h + moe(h))
        logits = self.lm_head(h)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)) \
            if labels is not None else logits


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
        if isinstance(mod, nn.Linear) and (leaf.startswith('q') or leaf.startswith('k')
                                           or leaf.startswith('v') or leaf.startswith('o')
                                           or name.startswith('mix.')):
            parent = model.get_submodule(name.rpartition('.')[0])
            setattr(parent, leaf, LoRALinear(mod))
            n += 1
    return n


def part_b(variant: str, steps: int = 3):
    torch.manual_seed(0)
    model = ToyMoEModel()
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    if variant == 'fp32':
        n_lora = wrap_attention_lora(model)
    else:
        load_kernel()
        quantize_model(model, 3, 64, r=8, use_kernel=True)
        n_lora = 0
        for name, mod in model.named_modules():
            if isinstance(mod, QuantLinear):
                # only the attention-like 'mix' adapters are trainable; experts/gate frozen
                trainable = name.startswith('mix.')
                mod.lora_A.requires_grad_(trainable)
                mod.lora_B.requires_grad_(trainable)
                if trainable:
                    n_lora += 1
    trainable = [p for p in model.parameters() if p.requires_grad]
    ids = torch.randint(1, model.config.vocab_size, (1, 64))
    usage = count_expert_usage(model, ids)
    labels = ids.clone()
    opt = bnb.optim.AdamW8bit(trainable, lr=1e-4)
    times, losses = [], []
    for step in range(steps + 1):
        t0 = time.perf_counter()
        loss = model(input_ids=ids, labels=labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step:
            times.append(time.perf_counter() - t0)
            losses.append(float(loss.detach()))
    med = sorted(times)[len(times) // 2] * 1e3
    active = 3 * 512 * 1024 * 2 + 4 * 512 * 512 + 2 * 2000 * 512 + 4 * 512 * 8
    print('  active params/token ~= %.2fM (total %.2fM)' % (active / 1e6,
          sum(p.numel() for p in model.parameters()) / 1e6))
    print('=== B[%s]: params %.2fM | LoRA %d | trainable %.3fM | weight-mem %.3f GB | step %.1f ms | rss %.2f GB'
          % (variant, sum(p.numel() for p in model.parameters()) / 1e6, n_lora,
             sum(p.numel() for p in trainable) / 1e6, model_mem_gb(model), med, rss_gb()))
    print('  expert usage:', {k: v for k, v in list(usage.items())[:3]})
    print('  loss %.4f -> %.4f' % (losses[0], losses[-1]))
    print('RESULT scaled_%s step_ms=%.1f weight_mem=%.3f rss=%.2f loss_first=%.4f loss_last=%.4f'
          % (variant, med, model_mem_gb(model), rss_gb(), losses[0], losses[-1]))


def main():
    part_a()
    print()
    part_b('fp32', steps=3)
    print()
    part_b('3bit', steps=3)
    print()
    part_c()
    print('\nMOE BENCH DONE')


if __name__ == '__main__':
    main()
