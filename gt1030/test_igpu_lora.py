# -*- coding: utf-8 -*-
"""test_igpu_lora.py — frozen base + LoRA adapters through igpu_fx vs torch.

python gt1030/test_igpu_lora.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import pyopencl as cl
import torch
import torch.nn as nn
import torch.nn.functional as F

import gt1030.igpu_fx as FX
from gt1030.igpu_lora import lora_parameters, wrap_lora

SCALE = 65536.0


class LinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(16, 32)
        self.f2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.f2(F.gelu(self.f1(x)))


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        return self.c(x)


def run(model, x, r=4, steps=12, lr=2e-4, conv=False):
    nwrap = wrap_lora(model, r=r, conv=conv)
    ref = type(model)()
    wrap_lora(ref, r=r, conv=conv)
    ref.load_state_dict(model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)
    for p in lora_parameters(ref):
        p.requires_grad_(True)

    ex = FX.Executor(
        model,
        x,
        state_bits=32,
        arena_mb=1200,
        freeze_base=True,
        trainable_names=["lora"],
    )
    ex.auto_zero_grads = True
    ex.sync_from_torch()
    tgt = torch.randn_like(ref(x)).detach()
    opt = torch.optim.Adam(lora_parameters(ref), lr=lr, eps=1e-8)

    dev, cpu = [], []
    n = int(np.prod(tgt.shape))
    for t in range(1, steps + 1):
        ex.zero_grads()
        y = ex.forward({"x": x.numpy()})
        yv = ex.dev.down(y.buf, y.shape).astype(np.float32)
        loss = float(np.mean((yv - tgt.numpy()) ** 2))
        dy = (2.0 / n) * (yv - tgt.numpy()) * SCALE
        dyb = ex._cached("lora.dy", (n,))
        cl.enqueue_copy(ex.dev.q, dyb, np.ascontiguousarray(dy.reshape(-1), np.float16))
        ex.backward(dyb, y.shape)
        ex.step(lr, t, eps=1e-8 * SCALE)
        dev.append(loss)

        opt.zero_grad()
        loss_t = ((ref(x) - tgt) ** 2).mean()
        loss_t.backward()
        opt.step()
        cpu.append(float(loss_t.detach()))

    rel = abs(dev[-1] - cpu[-1]) / max(abs(cpu[-1]), 1e-12)
    print(
        "  %s r=%d wraps=%d | device %.6f -> %.6f | torch %.6f -> %.6f | rel %.2e"
        % (type(model).__name__, r, nwrap, dev[0], dev[-1], cpu[0], cpu[-1], rel)
    )
    return rel < 5e-2


def main() -> int:
    torch.manual_seed(0)
    ok = True
    ok &= run(LinNet(), torch.randn(8, 16), conv=False)
    ok &= run(ConvNet(), torch.randn(2, 3, 16, 16), conv=True)
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
