# -*- coding: utf-8 -*-
"""profile_step_extra.py — follow-ups to profile_real_step.

1. Thread count tuning on the real Qwen3.5-0.8B LoRA step (4/5/6 threads).
2. How much of the step is GDN? Wrap bitsandbytes' fused kernel to count calls/time, and
   compare the same model with and without patch_transformers (patched vs eager fallback).
3. At a longer sequence (--long-seq, default 512) where the recurrent GDN cost should show.

Usage: python profile_step_extra.py [--seq 64] [--long-seq 512] [--skip-long]
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time

import torch

sys.path.insert(0, r"D:\work\bitsandbytes-CPU\bitsandbytes")


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def build_lora(model, r=8, alpha=16):
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base):
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

    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(model, name, LoRALinear(child))
            n += 1
        else:
            n += build_lora(child, r, alpha)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\deepsleep\_test\models\qwen3.5-0.8B")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--long-seq", type=int, default=512)
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()

    from transformers import Qwen3_5ForConditionalGeneration
    import bitsandbytes as bnb
    from bitsandbytes import gdn_cpu

    torch.set_num_threads(6)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, low_cpu_mem_usage=True)
    model.train()
    freeze_all(model)
    build_lora(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(params, lr=1e-4)
    hidden = getattr(model.config, "text_config", model.config).hidden_size

    def step(ids, target):
        t0 = time.perf_counter()
        h = model.model(input_ids=ids).last_hidden_state
        loss = torch.nn.functional.mse_loss(h, target)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return time.perf_counter() - t0

    ids = torch.randint(1, 1000, (1, args.seq))
    target = torch.zeros(1, args.seq, hidden)
    step(ids, target)  # warmup

    print("=== 1) thread scaling (seq=%d, median of 3 after warmup) ===" % args.seq)
    results = {}
    for th in (2, 3, 4, 5, 6):
        torch.set_num_threads(th)
        step(ids, target)  # warmup at this thread count
        t = st.median([step(ids, target) for _ in range(3)])
        results[th] = t
        print("  threads=%d  %.0f ms/step" % (th, t * 1e3))
    best = min(results, key=results.get)
    print("  best: threads=%d (%.0f ms, %.0f%% of 6-thread time)"
          % (best, results[best] * 1e3, 100.0 * results[best] / results[6]))
    torch.set_num_threads(6)

    # ---------------------------------------------------------------- 2) GDN share
    print("\n=== 2) GDN share (instrumented fused kernel) ===")
    from bitsandbytes.gdn_cpu import patch_transformers
    # remember the original module globals so the long-seq section can un-patch cleanly
    import transformers.models.qwen3_next.modeling_qwen3_next as mnext
    import transformers.models.qwen3_5.modeling_qwen3_5 as m35
    saved = []
    for mod in (mnext, m35):
        for attr in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
            fn = getattr(mod, attr, None)
            if fn is not None:
                saved.append((mod, attr, fn))
    patched = patch_transformers()
    stats = {"calls": 0, "secs": 0.0}
    orig_fused = gdn_cpu.fused_recurrent_gated_delta_rule

    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_fused(*a, **kw)
        finally:
            stats["calls"] += 1
            stats["secs"] += time.perf_counter() - t0

    gdn_cpu.fused_recurrent_gated_delta_rule = timed
    try:
        step(ids, target)  # warmup
        per = [step(ids, target) for _ in range(3)]
    finally:
        gdn_cpu.fused_recurrent_gated_delta_rule = orig_fused
    med = st.median(per)
    print("  patch applied=%s" % patched)
    print("  fused GDN calls/step: %d | GDN time %.1f ms/step (%.1f%% of %.0f ms)"
          % (stats["calls"] / 4, stats["secs"] / 4 * 1e3,
             100.0 * stats["secs"] / 4 / med, med * 1e3))

    # ---------------------------------------------------------------- 3) long seq
    if not args.skip_long:
        print("\n=== 3) long sequence: unpatched vs patched (seq=%d) ===" % args.long_seq)
        ids_l = torch.randint(1, 1000, (1, args.long_seq))
        target_l = torch.zeros(1, args.long_seq, hidden)

        # unpatched: restore the original transformers globals captured above
        for mod, attr, fn in saved:
            setattr(mod, attr, fn)
        step(ids_l, target_l)  # warmup
        unpatched = step(ids_l, target_l)

        patched = patch_transformers()  # re-patch (originals have no marker -> patched again)
        step(ids_l, target_l)
        patched_t = step(ids_l, target_l)
        print("  unpatched %.0f ms | patched %.0f ms (%.2fx) | patch=%s"
              % (unpatched * 1e3, patched_t * 1e3, unpatched / patched_t, patched))

    print("\nEXTRA PROFILE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
