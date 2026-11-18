# -*- coding: utf-8 -*-
"""bench_unet_hybrid.py — 三种执行模式的**实测**对比：全 CPU / 全核显 / 混合（逐层挑快的）。

为什么这样做：整条 UNet 的设备路径数值还没跑通（A.16），但每一层 conv 都是真执行、
真搬运数据，所以"逐层挑选后端"的总耗时可实测，而不是靠加法外推。

判定规则（来自 A.17.9 的形状表）：先各测一次每层 CPU/GPU 时间（重复 2 次取小），
`gpu/cpu < 0.95` 才放核显；`conv_out` 这类极低通道层强制留 CPU。

    python gt1030/bench_unet_hybrid.py --latent 64
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.bench_unet_conv_igpu import unet_conv_shapes  # noqa: E402
from gt1030.igpu_conv import Conv2d  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--thresh", type=float, default=0.95)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    shapes = unet_conv_shapes(a.latent)
    dev = CTX.DevCtx(arena_mb=3000, name="hybrid")
    dev.activate()

    rows = []
    for (name, cin, cout, H, W, stride) in shapes:
        if H * W == 0:
            continue
        conv = torch.nn.Conv2d(cin, cout, 3, stride, 1, bias=True).eval()
        x = torch.randn(1, cin, H, W)
        w = conv.weight.detach().numpy()
        b = conv.bias.detach().numpy()
        with torch.no_grad():
            conv(x)
            best_cpu = 1e9
            for _ in range(a.reps):
                t0 = time.perf_counter(); conv(x); best_cpu = min(best_cpu, time.perf_counter() - t0)

        dconv = Conv2d(dev, 1, cin, H, W, cout, 3, 3, stride, stride, 1, 1, bias=True)
        xb = dev.up(np.asarray(x.numpy(), np.float16))
        wb = dev.up(np.ascontiguousarray(w.astype(np.float16)))
        bb = dev.up(np.ascontiguousarray(b.astype(np.float16)))
        dconv.forward(xb, wb, bb); dev.q.finish()
        best_gpu = 1e9
        for _ in range(a.reps):
            t0 = time.perf_counter(); dconv.forward(xb, wb, bb); dev.q.finish()
            best_gpu = min(best_gpu, time.perf_counter() - t0)
        ratio = best_gpu / best_cpu
        # 决定后端：极低通道/输出层有惩罚，强制 CPU
        use_gpu = (ratio < a.thresh) and cout >= 64
        rows.append((name, best_cpu, best_gpu, ratio, use_gpu))

    t_all_cpu = sum(r[1] for r in rows)
    t_all_gpu = sum(r[2] for r in rows)
    t_hybrid = sum((r[2] if r[4] else r[1]) for r in rows)
    n_gpu = sum(1 for r in rows if r[4])

    print("  层数 %d · 放核显 %d 层 · 阈值 %.2f" % (len(rows), n_gpu, a.thresh))
    print("  全 CPU   %7.1f ms   (1.00x)" % (1000 * t_all_cpu))
    print("  全核显   %7.1f ms   (%.2fx)" % (1000 * t_all_gpu, t_all_gpu / t_all_cpu))
    print("  **混合** %7.1f ms   (%.2fx)  ← 逐层挑快的那条" % (1000 * t_hybrid, t_hybrid / t_all_cpu))
    worst = sorted(rows, key=lambda r: -r[3])[:3]
    print("  最差三层（留 CPU）:", ", ".join("%s %.2fx" % (r[0], r[3]) for r in worst))
    best = sorted(rows, key=lambda r: r[3])[:3]
    print("  最好三层（交核显）:", ", ".join("%s %.2fx" % (r[0], r[3]) for r in best))
    print("  RESULT 混合最快" if t_hybrid < min(t_all_cpu, t_all_gpu) else "  RESULT 全 CPU 最快")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
