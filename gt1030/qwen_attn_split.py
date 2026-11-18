# -*- coding: utf-8 -*-
"""qwen_attn_split.py — 量 1080p 尺度下"注意力 vs GEMM"的时间占比。

做法：monkeypatch torch.nn.functional.scaled_dot_product_attention 计时，
跑一次完整 forward，把注意力耗时与总耗时相减得到 GEMM/其它部分。
用法：python qwen_attn_split.py 8160
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F
import psutil

from qwen4bit_loader import load_transformer

FULL = r"D:\work\qwen_env\Qwen-Image-2.1"
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

    orig = F.scaled_dot_product_attention

    def timed(*a, **kw):
        timed.t0 = time.perf_counter()
        out = orig(*a, **kw)
        timed.total += time.perf_counter() - timed.t0
        timed.calls += 1
        return out

    timed.total = 0.0
    timed.calls = 0
    F.scaled_dot_product_attention = timed
    for mod in tr.modules():
        if hasattr(mod, 'scaled_dot_product_attention'):
            mod.scaled_dot_product_attention = timed
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            tr(hidden_states=lat, timestep=ts, encoder_hidden_states=pe,
               encoder_hidden_states_mask=pm, img_shapes=shapes, img_mask=img_mask,
               return_dict=False)
        total = time.perf_counter() - t0
    finally:
        F.scaled_dot_product_attention = orig
        for mod in tr.modules():
            if getattr(mod, 'scaled_dot_product_attention', None) is timed:
                mod.scaled_dot_product_attention = orig

    attn = timed.total
    print(f"  seq={seq} | 总 {total:.1f} s | 注意力 {attn:.1f} s（{attn/total*100:.1f}%，{timed.calls} 次调用）"
          f" | 其余(GEMM等) {total-attn:.1f} s")
    # 注意力理论 FLOPs（joint：q 长度 seq+T，kv seq）
    T = pe.shape[1]
    flops = 4 * seq * seq * 128 * 32
    print(f"  注意力等效吞吐 {flops/attn/1e9:.0f} GFLOPS | 其余 {2*7.12e9*seq/(total-attn)/1e9:.0f} GFLOPS")
    print("RESULT " + str(dict(seq=seq, total=round(total, 1), attn=round(attn, 1),
                               share=round(attn / total, 3), calls=timed.calls)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2304)
