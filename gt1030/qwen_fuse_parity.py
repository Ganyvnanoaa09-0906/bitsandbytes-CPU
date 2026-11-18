# -*- coding: utf-8 -*-
"""qwen_fuse_parity.py — 校验"LoRA 融合进 NF4"与"hook 增量"是否等价。

同一份模型与输入：
  1) hook 方式加 LoRA → forward 得 A
  2) 移除 hook，把 LoRA 融进权重 → forward 得 B
比较 relL2。若显著大于 1e-2，说明融合有 bug（重量化误差或尺度错）。
"""
from __future__ import annotations

import sys
import torch

from qwen4bit_loader import load_transformer
from qwen_fuse_lora import fuse
from qwen_lightning_bench import LoRAInjector, LORA

EMB = r"D:\work\research\bench_results\qwen_prompt_embeds.pt"


def main(seq=1024):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tr = load_transformer()
    d = torch.load(EMB, map_location="cpu")
    pe = d["prompt_embeds"].float()
    pm = d.get("prompt_embeds_mask")
    base = d.get("image_pad_mask", torch.zeros(1, pe.shape[1], dtype=torch.bool)).bool()
    img_mask = torch.cat([base, torch.ones(1, seq // 4, dtype=torch.bool)], dim=1)
    lat = torch.randn(1, seq, 64)
    side = int(round(seq ** 0.5))
    shapes = [[(1, side, side)]]
    ts = torch.tensor([500.0])

    def fwd():
        with torch.no_grad():
            return tr(hidden_states=lat, timestep=ts, encoder_hidden_states=pe,
                      encoder_hidden_states_mask=pm, img_shapes=shapes, img_mask=img_mask,
                      return_dict=False)[0].float()

    inj = LoRAInjector(tr, LORA, scale=1.0)
    A = fwd().clone()
    inj.remove()
    fuse(tr, LORA, scale=1.0)
    B = fwd()
    rel = float((A - B).norm() / A.norm())
    print(f"  seq={seq} | hook vs 融合: relL2 {rel:.3e} | max|Δ| {float((A-B).abs().max()):.3e}")
    print(f"  hook  输出 std {float(A.std()):.5f} | 融合输出 std {float(B.std()):.5f}")
    print("RESULT " + str(dict(seq=seq, rel=rel)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1024)
