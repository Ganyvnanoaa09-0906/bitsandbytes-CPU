# -*- coding: utf-8 -*-
"""profile_real_step.py (v2) — clean breakdown of a real CPU LoRA training step.

Fixes vs v1:
  * freeze EVERY base parameter first, then make only lora_A/lora_B trainable (real LoRA:
    ~5M trainable, not 548M);
  * pre-create the loss target (v1 allocated a zero tensor every step, polluting fill_);
  * profiler runs with with_stack and prints the Python stacks that own the biggest fill_/mul
    operators, so we can find removable allocations;
  * optional --gdn-compare at a longer sequence where the GDN recurrent cost matters.

Usage:
    python profile_real_step.py --seq 64
    python profile_real_step.py --seq 256 --gdn-compare
"""
from __future__ import annotations
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

import argparse
import statistics as st
import sys
import time

import torch


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
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(model, name, LoRALinear(child))
            count += 1
        else:
            count += build_lora(child, r, alpha)
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\deepsleep\_test\models\qwen3.5-0.8B")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--gdn-compare", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from transformers import Qwen3_5ForConditionalGeneration
    import bitsandbytes as bnb

    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, low_cpu_mem_usage=True)
    model.train()
    freeze_all(model)
    n_lora = build_lora(model)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(lora_params, lr=1e-4)
    print("loaded %.1fs | LoRA linears %d | trainable %.2fM"
          % (time.time() - t0, n_lora, sum(p.numel() for p in lora_params) / 1e6))

    ids = torch.randint(1, 1000, (1, args.seq))
    target = torch.zeros(1, args.seq, model.config.text_config.hidden_size
                         if hasattr(model.config, "text_config") else 1024)

    def one_step():
        t_fwd = time.perf_counter()
        h = model.model(input_ids=ids).last_hidden_state
        t_fwd1 = time.perf_counter()
        loss = torch.nn.functional.mse_loss(h, target)
        loss.backward()
        t_bwd1 = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        t_opt = time.perf_counter()
        return (t_fwd1 - t_fwd, t_bwd1 - t_fwd1, t_opt - t_bwd1, float(loss.detach()))

    times = []
    for i in range(args.steps + 1):
        t = one_step()
        if i:
            times.append(t)
    fwd = st.median(x[0] for x in times)
    bwd = st.median(x[1] for x in times)
    stept = st.median(x[2] for x in times)
    print("step breakdown: forward %.0f ms | backward %.0f ms | optimizer %.1f ms | total %.0f ms"
          % (fwd * 1e3, bwd * 1e3, stept * 1e3, (fwd + bwd + stept) * 1e3))
    print("loss %.6f" % times[-1][3])
    try:
        import psutil
        print("rss: %.0f MB" % (psutil.Process().memory_info().rss / (1 << 20)))
    except Exception:
        pass

    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        one_step()
    print("\n=== top self CPU time ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15,
                                    max_name_column_width=50))

    # Which Python stacks own the biggest fill_/copy_/mul calls?
    print("\n=== stacks for fill_ / mul (top repeated call sites) ===")
    seen = {}
    for evt in prof.events():
        if evt.key not in ("aten::fill_", "aten::mul", "aten::copy_", "aten::bmm"):
            continue
        stack = getattr(evt, "stack", None) or []
        # keep the deepest few frames that are not torch internals
        frames = []
        for sym in stack:
            fn = getattr(sym, "name", str(sym))
            frames.append(fn)
        key = (evt.key, tuple(frames[-3:]))
        seen[key] = seen.get(key, 0) + 1
    for (key, frames), n in sorted(seen.items(), key=lambda kv: -kv[1])[:8]:
        print("  %-14s x%-6d %s" % (key, n, " <- ".join(frames)))

    # patched vs unpatched GDN at this sequence length
    if args.gdn_compare:
        try:
            from bitsandbytes.gdn_cpu import patch_transformers
            base_t = sum(one_step()[:3])
            patched = patch_transformers()
            pat = [sum(one_step()[:3]) for _ in range(2)]
            print("\nGDN patch applied=%s | unpatched %.0f ms | patched %.0f ms (%.2fx)"
                  % (patched, base_t * 1e3, st.median(pat) * 1e3,
                     base_t / st.median(pat)))
        except Exception as exc:
            print("GDN compare skipped:", exc)

    print("\nPROFILE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
