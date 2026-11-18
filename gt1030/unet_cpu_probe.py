# -*- coding: utf-8 -*-
"""unet_cpu_probe.py — run the real SD1.5 UNet forward once on CPU.

This is a feasibility probe, not the iGPU path: it establishes that the real
model/weights execute and measures host RSS + latency. The iGPU memory risk is
measured separately by the attention probe.

    python gt1030/unet_cpu_probe.py
"""
from __future__ import annotations

import gc
import os
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import psutil
import torch

UNET = r"D:\work\textmodel\_sd15_combined\unet"


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1048576


def main() -> int:
    from diffusers import UNet2DConditionModel

    print("  rss before load %.0f MB" % rss_mb())
    t0 = time.time()
    unet = UNet2DConditionModel.from_pretrained(
        UNET, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    unet.eval()
    load_s = time.time() - t0
    n = sum(p.numel() for p in unet.parameters())
    print("  loaded %.3f B params · %.1fs · rss %.0f MB" %
          (n / 1e9, load_s, rss_mb()))

    x = torch.randn(1, 4, 64, 64)
    ts = torch.tensor([500], dtype=torch.long)
    txt = torch.randn(1, 77, 768)
    t0 = time.time()
    with torch.no_grad():
        y = unet(x, ts, encoder_hidden_states=txt).sample
    fwd_s = time.time() - t0
    print("  forward %.2fs · out %s · rss %.0f MB" %
          (fwd_s, tuple(y.shape), rss_mb()))
    print("  finite=%s" % bool(torch.isfinite(y).all()))
    del unet, y, x, txt
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
