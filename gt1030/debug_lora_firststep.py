# -*- coding: utf-8 -*-
"""debug_lora_firststep.py — LoRA adapter 首步 Δw 与方向余弦（device vs torch）。

    python gt1030/debug_lora_firststep.py [--noise 0.1] [--lr 2e-4]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\gt1030")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402
import torch  # noqa: E402

import gt1030.igpu_fx as FX  # noqa: E402
from gt1030.igpu_lora import LoRAConv2d, LoRALinear, lora_parameters  # noqa: E402
from train_real_vae import build_encoder  # noqa: E402

SCALE = 65536.0


def wrap_all(mod, r):
    n = 0
    for name, child in list(mod.named_children()):
        if isinstance(child, LoRALinear) or isinstance(child, LoRAConv2d):
            continue
        if isinstance(child, torch.nn.Conv2d):
            setattr(mod, name, LoRAConv2d(child, r))
            n += 1
        else:
            n += wrap_all(child, r)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(0)
    enc = build_encoder(r"D:\work\textmodel\_sd15_combined\vae", 64)
    n = wrap_all(enc, a.r)
    lora_names = [nm for nm, _ in enc.named_parameters() if ".lora_" in nm]
    prefixes = sorted({nm.rsplit(".", 1)[0] for nm in lora_names})
    print("  LoRA r=%d 包裹 %d 层 · adapter 参数 %d 个" % (a.r, n, len(lora_names)))

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        tgt = enc(x) + torch.randn_like(enc(x)) * a.noise

    # ---- 设备一步 ----
    ex = FX.Executor(enc, x, state_bits=32, arena_mb=2500,
                     freeze_base=True, trainable_names=prefixes)
    ex.auto_zero_grads = True
    ex.sync_from_torch()
    snap = {}
    for nm in lora_names:
        p = ex.params.get(nm)
        if p is None or getattr(p, "frozen", False):
            continue
        snap[nm] = ex.dev.down(p.buf, p.shape).astype(np.float32).copy()
    ex.zero_grads()
    y = ex.forward({"x": x.numpy()})
    nnum = int(np.prod(y.shape))
    yv = ex.dev.down(y.buf, y.shape).astype(np.float32)
    dy = (2.0 / nnum) * (yv - tgt.numpy()) * SCALE
    dyb = ex._cached("lora.dy", (nnum,))
    cl.enqueue_copy(ex.dev.q, dyb, np.ascontiguousarray(dy.reshape(-1), np.float16))
    ex.backward(dyb, y.shape)
    ex.step(a.lr, 1, eps=1e-8 * SCALE)
    ex.dev.q.finish()

    # ---- torch 一步（同一模块、同权重）----
    ref = build_encoder(r"D:\work\textmodel\_sd15_combined\vae", 64)
    n2 = wrap_all(ref, a.r)
    ref.load_state_dict(enc.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)
    for p in lora_parameters(ref):
        p.requires_grad_(True)
    opt = torch.optim.Adam(lora_parameters(ref), lr=a.lr, eps=1e-8)
    opt.zero_grad()
    ((ref(x) - tgt) ** 2).mean().backward()
    opt.step()
    tmap = dict(ref.named_parameters())

    # ---- 逐参数比 Δ ----
    dots, norms, cosines = 0.0, 0.0, []
    worst = None
    for nm in snap:
        p = ex.params[nm]
        after = ex.dev.down(p.buf, p.shape).astype(np.float32)
        d_dev = (after - snap[nm]).reshape(-1)
        d_ref = tmap[nm].detach().numpy().reshape(-1)
        nd, nr = np.linalg.norm(d_dev), np.linalg.norm(d_ref)
        if nr < 1e-12:
            continue
        c = float(np.dot(d_dev, d_ref) / (nd * nr + 1e-30))
        cosines.append(c)
        dots += float(np.dot(d_dev, d_ref))
        norms += nd * nr
        if worst is None or c < worst[1]:
            worst = (nm, c, nd / nr)
    print("  参与比较的 adapter 参数: %d" % len(cosines))
    print("  整体 cos = %.4f · 平均 cos = %.4f · 最小 cos = %.4f (%s)"
          % (dots / (norms + 1e-30), float(np.mean(cosines)), worst[1], worst[0][:48]))
    print("  最小 cos 那个参数的 |Δdev|/|Δtorch| = %.3f" % worst[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
