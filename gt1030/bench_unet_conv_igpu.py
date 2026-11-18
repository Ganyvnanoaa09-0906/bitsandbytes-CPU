# -*- coding: utf-8 -*-
"""bench_unet_conv_igpu.py — SD1.5 UNet 的**卷积形状级**对照：CPU vs 核显，并做端到端投影。

为什么不直接跑整条 UNet：A.16 的设备路径还没跑通数值（输出全零，诊断中）。
但只要逐形状测准，就能把"整模型上核显会不会更快"用**算子组成**算出来——
而且这正是 A.16.59 里 conv 对拍已经验证过的路径（rel 5e-3）。

    python gt1030/bench_unet_conv_igpu.py --latent 64
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.igpu_conv import Conv2d, K as CVK  # noqa: E402

UNET = r"D:\work\textmodel\_sd15_combined\unet"


def unet_conv_shapes(latent=64):
    """按 SD1.5 UNet 的真实结构生成 (名字, in, out, H, W, stride) 列表。"""
    from diffusers import UNet2DConditionModel

    u = UNet2DConditionModel.from_pretrained(UNET)
    shapes = []
    H = W = latent
    c = u.conv_in.out_channels
    shapes.append(("conv_in", 4, c, H, W, 1))
    chans = [u.conv_in.out_channels] + list(u.config.block_out_channels)
    for i, db in enumerate(u.down_blocks):
        cin = u.config.block_out_channels[i]
        for j, res in enumerate(db.resnets):
            cout = res.conv1.out_channels
            shapes.append(("down%d.res%d.c1" % (i, j), cin, cout, H, W, 1))
            shapes.append(("down%d.res%d.c2" % (i, j), cout, cout, H, W, 1))
            if res.conv_shortcut is not None:
                shapes.append(("down%d.res%d.sc" % (i, j), cin, cout, H, W, 1))
            cin = cout
        if getattr(db, "downsamplers", None):
            cout = db.downsamplers[0].conv.out_channels
            shapes.append(("down%d.ds" % i, cin, cout, H, W, 2))
            H, W = (H + 1) // 2, (W + 1) // 2
            cin = cout
    # mid
    mid = u.mid_block
    for j, res in enumerate(mid.resnets):
        cin = res.conv1.in_channels
        cout = res.conv1.out_channels
        shapes.append(("mid.res%d.c1" % j, cin, cout, H, W, 1))
        shapes.append(("mid.res%d.c2" % j, cout, cout, H, W, 1))
    # up（跳过 upsample 与 attention，只算 resnet 卷积；H/W 逐块翻倍）
    for i, ub in enumerate(u.up_blocks):
        for j, res in enumerate(ub.resnets):
            cin = res.conv1.in_channels
            cout = res.conv1.out_channels
            shapes.append(("up%d.res%d.c1" % (i, j), cin, cout, H, W, 1))
            shapes.append(("up%d.res%d.c2" % (i, j), cout, cout, H, W, 1))
        if getattr(ub, "upsamplers", None):
            cout = ub.upsamplers[0].conv.out_channels
            H, W = H * 2, W * 2
            shapes.append(("up%d.up" % i, cout, cout, H, W, 1))
    shapes.append(("conv_out", u.conv_out.in_channels, u.conv_out.out_channels, H, W, 1))
    return shapes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    shapes = unet_conv_shapes(a.latent)
    dev = CTX.DevCtx(arena_mb=3000, name="unetconv")
    dev.activate()

    print("  %-18s %5s %5s %4s %4s %-2s %9s %9s %7s" %
          ("op", "in", "out", "H", "W", "s", "CPU ms", "GPU ms", "GPU/CPU"))
    t_cpu = t_gpu = 0.0
    for (name, cin, cout, H, W, stride) in shapes:
        if H * W == 0:
            continue
        x = torch.randn(1, cin, H, W)
        conv = torch.nn.Conv2d(cin, cout, 3, stride, 1, bias=True).eval()
        w = conv.weight.detach().numpy()
        b = conv.bias.detach().numpy()
        with torch.no_grad():
            conv(x)  # warm
            t0 = time.perf_counter()
            for _ in range(a.reps):
                conv(x)
            cpu_ms = 1e3 * (time.perf_counter() - t0) / a.reps

        dconv = Conv2d(dev, 1, cin, H, W, cout, 3, 3, stride, stride, 1, 1, bias=True)
        xb = dev.up(np.asarray(x.numpy(), np.float16))
        wb = dev.up(np.ascontiguousarray(w.astype(np.float16)))
        bb = dev.up(np.ascontiguousarray(b.astype(np.float16)))
        dconv.forward(xb, wb, bb)
        dev.q.finish()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            dconv.forward(xb, wb, bb)
        dev.q.finish()
        gpu_ms = 1e3 * (time.perf_counter() - t0) / a.reps
        t_cpu += cpu_ms
        t_gpu += gpu_ms
        print("  %-18s %5d %5d %4d %4d %-2d %9.2f %9.2f %7.2f"
              % (name, cin, cout, H, W, stride, cpu_ms, gpu_ms, gpu_ms / cpu_ms))

    print("\n  合计（仅卷积，%d 个）: CPU %.1f ms · 核显 %.1f ms · 比例 %.2fx"
          % (len(shapes), t_cpu, t_gpu, t_gpu / t_cpu))
    print("  RESULT " + ("GPU 更快" if t_gpu < t_cpu else "CPU 更快"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
