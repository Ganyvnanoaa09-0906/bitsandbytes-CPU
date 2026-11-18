# -*- coding: utf-8 -*-
"""profile_lora_overhead.py — 量 LoRA adapter 的步时开销与它的缩放规律。

    python gt1030/profile_lora_overhead.py --steps 5

只读测量，不改任何训练脚本。回答一个问题：
    adapter 的开销是**线性于 r**（FLOPs 受限）还是**几乎不随 r 变**（launch 受限）？
  · 若 ms(r=4) - ms(r=0) 与 r 近似成正比 ⇒ FLOPs 受限 ⇒ 减层/降 r 最有效；
  · 若 r=2 与 r=4 的步时接近 ⇒ 固定开销（launch/调度）占主导 ⇒ 才值得做融合 kernel。
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

import gt1030.igpu_fx as FX  # noqa: E402
from gt1030.igpu_lora import wrap_lora  # noqa: E402

SCALE = 65536.0
VAE = r"D:\work\textmodel\_sd15_combined\vae"


def one_case(r: int, steps: int, noise: float = 0.5):
    from train_real_vae import build_encoder

    torch.manual_seed(0)
    enc = build_encoder(VAE, 64)
    nwrap, adapter = 0, 0
    if r > 0:
        nwrap = wrap_lora(enc, r=r, conv=True)
        adapter = sum(p.numel() for nm, p in enc.named_parameters() if ".lora_" in nm)
    names = [n for n, _ in enc.named_parameters() if ".lora_" in n]
    prefixes = sorted({n.rsplit(".", 1)[0] for n in names}) if r > 0 else None

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        tgt = enc(x) + torch.randn_like(enc(x)) * noise

    ex = FX.Executor(enc, x, state_bits=8, arena_mb=2500,
                     freeze_base=bool(r), trainable_names=prefixes)
    ex.auto_zero_grads = True
    ex.sync_from_torch()

    ms, loss = [], 0.0
    for t in range(1, steps + 1):
        s = time.time()
        ex.zero_grads()
        y = ex.forward({"x": x.numpy()})
        n = int(np.prod(y.shape))
        yv = ex.dev.down(y.buf, y.shape).astype(np.float32)
        loss = float(np.mean((yv - tgt.numpy()) ** 2))
        dy = (2.0 / n) * (yv - tgt.numpy()) * SCALE
        dyb = ex._cached("prof.dy", (n,))
        cl.enqueue_copy(ex.dev.q, dyb, np.ascontiguousarray(dy.reshape(-1), np.float16))
        ex.backward(dyb, y.shape)
        ex.step(2e-4, t, eps=1e-8 * SCALE)
        ex.q_finish() if hasattr(ex, "q_finish") else ex.dev.q.finish()
        ms.append(1000 * (time.time() - s))
    med = float(np.median(ms[1:])) if len(ms) > 1 else float(ms[0])
    return dict(r=r, wraps=nwrap, adapter=adapter, ms=med, loss=loss, arena=ex.dev._off / 1048576)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()

    rows = []
    for r in (0, 2, 4, 8):
        d = one_case(r, a.steps)
        rows.append(d)
        print("  r=%d  包裹 %3d 层  adapter %7.3f M  %7.1f ms/步  arena %.0f MB"
              % (d["r"], d["wraps"], d["adapter"] / 1e6, d["ms"], d["arena"]))

    base = rows[0]["ms"]
    print("\n  相对基线（r=0 = %.1f ms）的增量：" % base)
    for d in rows[1:]:
        delta = d["ms"] - base
        per_r = delta / d["r"] if d["r"] else 0
        print("  r=%d  +%6.1f ms  (+%.1f%%)  每单位 r: %+5.1f ms"
              % (d["r"], delta, 100 * delta / base, per_r))
    r2 = next(d for d in rows if d["r"] == 2)["ms"] - base
    r4 = next(d for d in rows if d["r"] == 4)["ms"] - base
    ratio = r4 / r2 if r2 > 1e-6 else float("inf")
    print("\n  (r=4 增量)/(r=2 增量) = %.2f" % ratio)
    if ratio < 1.4:
        print("  ⇒ 增量几乎不随 r 变 ⇒ **固定开销（launch/调度）主导**，融合 kernel 才有意义")
    elif ratio < 2.4:
        print("  ⇒ 接近线性于 r ⇒ **FLOPs 受限**，减层/降 r 最有效（融合 kernel 收益有限）")
    else:
        print("  ⇒ 超线性 ⇒ 可能存在阈值效应（缓存/占用率），需要逐层拆解")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
