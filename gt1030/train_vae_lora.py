# -*- coding: utf-8 -*-
"""train_vae_lora.py — 真实 SD1.5 VAE 编码器上的 **LoRA 训练**（底座冻结，只训 adapter）。

    python gt1030/train_vae_lora.py --steps 30 --r 4

判据：
  · adapter 参数量应远小于底座（打印占比）；
  · 参数显存应接近"冻结底座"（2 B/参数）+ adapter 的 9.6 B/参数；
  · loss 应单调下降（目标是编码输出 + 少量噪声，信噪比足够）。
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\gt1030")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import gt1030.igpu_fx as FX  # noqa: E402
from gt1030.igpu_lora import lora_parameters, wrap_lora  # noqa: E402
from train_real_vae import build_encoder  # noqa: E402

SCALE = 65536.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=r"D:\work\textmodel\_sd15_combined\vae")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.3)
    a = ap.parse_args()

    torch.manual_seed(0)
    enc = build_encoder(a.vae, 64)
    base_n = sum(p.numel() for p in enc.parameters())
    # 只包裹 stride==1 的卷积：LoRA 分支是 1x1 卷积，套在 stride=2 的采样层上会尺寸不匹配
    from gt1030.igpu_lora import LoRAConv2d

    nwrap = 0

    def walk(mod):
        nonlocal nwrap
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Conv2d) and tuple(child.stride) == (1, 1):
                setattr(mod, name, LoRAConv2d(child, a.r))
                nwrap += 1
            else:
                walk(child)

    walk(enc)
    print("  已包裹 stride=1 卷积 %d 层（采样层不包，LoRA 分支无法匹配 stride=2）" % nwrap)
    lora_names = [n for n, _ in enc.named_parameters() if ".lora_" in n]
    lora_n = sum(p.numel() for p in lora_parameters(enc))
    print("  LoRA r=%d：包裹 %d 层 · adapter %.2f M / 底座 %.1f M（%.2f%%）"
          % (a.r, nwrap, lora_n / 1e6, base_n / 1e6, 100 * lora_n / base_n))

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        tgt = enc(x) + torch.randn_like(enc(x)) * a.noise

    # 只有 adapter 可训：用模块名做白名单（LoRA 包装后的子模块名里带 lora_）
    prefixes = sorted({n.rsplit(".", 1)[0] for n in lora_names})
    ex = FX.Executor(enc, x, state_bits=8, arena_mb=2500,
                     freeze_base=True, trainable_names=prefixes)
    ex.auto_zero_grads = True
    ex.sync_from_torch()
    ntrain = sum(1 for p in ex.params.values() if not getattr(p, "frozen", False))
    nfr = sum(1 for p in ex.params.values() if getattr(p, "frozen", False))
    print("  设备参数：可训 %d 个 / 冻结 %d 个 · 参数显存 %.0f MB"
          % (ntrain, nfr, ex.dev._off / 1048576))

    losses, ms = [], []
    t0 = time.time()
    for t in range(1, a.steps + 1):
        s = time.time()
        ex.zero_grads()
        y = ex.forward({"x": x.numpy()})
        n = int(np.prod(y.shape))
        yv = ex.dev.down(y.buf, y.shape).astype(np.float32)
        losses.append(float(np.mean((yv - tgt.numpy()) ** 2)))
        dy = (2.0 / n) * (yv - tgt.numpy()) * SCALE
        dyb = ex._cached("lora.dy", (n,))
        cl.enqueue_copy(ex.dev.q, dyb, np.ascontiguousarray(dy.reshape(-1), np.float16))
        ex.backward(dyb, y.shape)
        ex.step(a.lr, t, eps=1e-8 * SCALE)
        ms.append(1000 * (time.time() - s))
    dt = time.time() - t0
    print("  step  loss")
    for i in (0, 1, a.steps // 2, a.steps - 1):
        if i < len(losses):
            print("  %4d  %.6f" % (i + 1, losses[i]))
    print("  末步 %.6f（下降 %.1f%%）· %.0f ms/步 · 总 %.1fs"
          % (losses[-1], 100 * (1 - losses[-1] / losses[0]), np.mean(ms), dt))
    ok = losses[-1] < losses[0] * 0.9
    print("RESULT " + ("PASS — 真实模型 LoRA 训练有效" if ok else "FAIL（loss 未明显下降）"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
