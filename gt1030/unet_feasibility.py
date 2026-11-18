# -*- coding: utf-8 -*-
"""unet_feasibility.py — read-only SD1.5 UNet inventory for the iGPU path.

Does not instantiate torch modules: reads config.json and the safetensors header,
then reports fp16 weight budget, attention block inventory, and the FullAttn
memory risk at 64x64 latent.

    python gt1030/unet_feasibility.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

UNET = Path(r"D:\work\textmodel\_sd15_combined\unet")
FP16 = 2


def main() -> int:
    cfg = json.loads((UNET / "config.json").read_text(encoding="utf-8"))
    model = UNET / "diffusion_pytorch_model.safetensors"
    from safetensors import safe_open

    total = 0
    by_top = {}
    attn1 = attn2 = 0
    with safe_open(str(model), framework="pt", device="cpu") as f:
        for k in f.keys():
            shape = tuple(f.get_slice(k).get_shape())
            n = int(np.prod(shape)) if shape else 1
            total += n
            top = k.split(".")[0]
            by_top[top] = by_top.get(top, 0) + n
            if ".attn1." in k:
                attn1 += 1
            if ".attn2." in k:
                attn2 += 1

    print("  UNet: %.3f B params" % (total / 1e9))
    print("  fp16 frozen weights: %.2f GB" % (total * FP16 / 1e9))
    print("  top-level params:")
    for k, n in sorted(by_top.items(), key=lambda x: -x[1])[:8]:
        print("    %-24s %.2f M" % (k, n / 1e6))

    print("  attention tensors: self(attn1)=%d cross(attn2)=%d" % (attn1, attn2))
    print("  config: sample=%s in=%s out=%s cross_dim=%s head_dim=%s" % (
        cfg.get("sample_size"), cfg.get("in_channels"), cfg.get("out_channels"),
        cfg.get("cross_attention_dim"), cfg.get("attention_head_dim")))

    # FullAttn currently allocates S/P/dP/dS at (N*T)^2 for self-attention.
    T = int(cfg.get("sample_size", 64)) ** 2
    N = 2
    rows = N * T
    per = rows * rows * FP16
    print("  FullAttn self-attn risk: T=%d N=%d rows=%d" % (T, N, rows))
    print("    each S/P/dP/dS = %.0f MB; four = %.0f MB"
          % (per / 1e6, 4 * per / 1e6))
    print("  cross-attn: K length is text tokens (77), not T; current FullAttn")
    print("    assumes square S and cannot run it as-is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
