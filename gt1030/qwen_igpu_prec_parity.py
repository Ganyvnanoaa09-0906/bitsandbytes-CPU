# -*- coding: utf-8 -*-
"""qwen_igpu_prec_parity.py — 同一输入下 iGPU fp32 vs fp16 混合的单 forward 误差（定位 1080p 花屏原因）。

用法：python qwen_igpu_prec_parity.py 2304
"""
from __future__ import annotations

import sys
import time

import torch

from qwen4bit_loader import load_transformer

EMB = r"D:\work\research\bench_results\qwen_prompt_embeds.pt"


def main(seq=2304):
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

    out_cpu = fwd()
    from qwen4bit_igpu_linear_fp32 import to_igpu_f32
    to_igpu_f32(tr, mode_fn=None)
    out_f32 = fwd()
    # 换成 fp16 包装：重新构造一份较麻烦，直接改已有 wrapper 的 mode 后重跑
    n16 = 0
    for mod in tr.modules():
        if hasattr(mod, 'mode') and getattr(mod, 'mode', None) == 'fp32':
            try:
                mod.mode = 'fp16'
                if hasattr(mod, '_prepare_buffers'):
                    mod._prepare_buffers()
                n16 += 1
            except Exception:
                pass
    print(f"  切到 fp16 的层数: {n16}", flush=True)
    out_f16 = fwd()
    for name, o in (('fp32', out_f32), ('fp16', out_f16)):
        rel = float((out_cpu - o).norm() / out_cpu.norm())
        print(f"  seq={seq} iGPU {name} vs CPU: relL2 {rel:.3e} | max|Δ| {float((out_cpu-o).abs().max()):.3e}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2304)
