# -*- coding: utf-8 -*-
"""bench_unet_e2e.py — 真正的端到端：整条 SD1.5 UNet 前向，CPU(torch fp32) vs 核显(fp16 常驻)。

A.17.10 那个 0.82× 是"逐层耗时之和"；现在整链数值修好了（rel 1.1e-2），
可以直接量**真实端到端**，这比层和更有说服力。
"""
import argparse
import statistics
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import torch

import gt1030.igpu_ctx as C
import gt1030.igpu_unet_full_dev as F
from diffusers import UNet2DConditionModel

UNET = r"D:\work\textmodel\_sd15_combined\unet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--cpu-reps", type=int, default=3)
    ap.add_argument("--gpu-reps", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(6)

    u = UNet2DConditionModel.from_pretrained(UNET).eval()
    torch.manual_seed(0)
    s = torch.randn(1, 4, a.latent, a.latent)
    enc = torch.randn(1, 77, u.config.cross_attention_dim)
    t = torch.tensor([500])

    # CPU 参考 + 计时
    cts = []
    with torch.no_grad():
        ref = u(s, t, encoder_hidden_states=enc).sample.numpy()
        for _ in range(a.cpu_reps):
            t0 = time.perf_counter()
            u(s, t, encoder_hidden_states=enc)
            cts.append(time.perf_counter() - t0)
    cpu_ms = statistics.median(cts) * 1e3

    dev = C.DevCtx(arena_mb=3900, name="e2e").activate()
    gts = []
    out = None
    for i in range(a.gpu_reps):
        t0 = time.perf_counter()
        out, shape = F.full_forward(dev, u, s.numpy(), np.array([500.0], np.float32), enc.numpy(),
                                    verbose=False)
        gts.append(time.perf_counter() - t0)
    gpu_ms = min(gts) * 1e3
    ov = dev.down(out, shape).astype(np.float32)
    rel = float(np.linalg.norm(ov - ref) / (np.linalg.norm(ref) + 1e-12))

    print("  latent=%d  CPU %.1f ms · 核显 %.1f ms · 比例 %.2fx · rel %.2e · 峰值 %.0f MB"
          % (a.latent, cpu_ms, gpu_ms, gpu_ms / cpu_ms, rel, F._mb(dev)))
    print("  RESULT " + ("核显更快" if gpu_ms < cpu_ms else "CPU 更快"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
