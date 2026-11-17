# -*- coding: utf-8 -*-
"""check_stack_ref.py — 2 层 stack（含注意力）5 步训练验收：device vs 手写 torch 参考。"""
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gt1030.igpu_fx as FX
import gt1030.igpu_graph as G

d, nh, hd, bt = 32, 4, 8, 16


class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = FX.AttnModule(nh, hd)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, d * 2, bias=False)
        self.f2 = nn.Linear(d * 2, d, bias=False)

    def forward(self, x):
        h = x + self.attn(self.n1(x))
        return h + self.f2(F.gelu(self.f1(self.n2(h))))


class S(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([B(), B()])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def tattn(mod, h):
    q = mod.q_proj(h).view(bt, nh, hd).transpose(0, 1)
    k = mod.k_proj(h).view(bt, nh, hd).transpose(0, 1)
    v = mod.v_proj(h).view(bt, nh, hd).transpose(0, 1)
    s = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
    mask = torch.tril(torch.ones(bt, bt, dtype=torch.bool))
    p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
    return mod.o_proj((p @ v).transpose(0, 1).reshape(bt, d))


def ref_forward(model, x):
    for blk in model.blocks:
        h = x + tattn(blk.attn, blk.n1(x))
        x = h + blk.f2(F.gelu(blk.f1(blk.n2(h))))
    return x


def main() -> int:
    torch.manual_seed(0)
    m = S()
    x = torch.randn(bt, d) * 0.3
    tgt = torch.randn(bt, d) * 0.1
    ex = FX.Executor(m, x)
    ex.sync_from_torch()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, eps=1e-8)
    scale = 65536.0
    yt = ex.dev.up(tgt.numpy().astype(np.float16))
    dy = ex.dev.buf(max(4096, bt * d * 2))
    worst = 0.0
    print("  step    device     torch        rel")
    for step in range(1, 6):
        ex.zero_grads()
        y = ex.forward(x.numpy())
        n = y.numel
        G._kernels(ex.dev)["mse_grad_h"](ex.dev.q, (n,), None, y.buf, yt, dy,
                                         np.int64(n), np.float32(1.0 / n), np.float32(scale))
        ex.dev.q.finish()
        ld = float(((ex.dev.down(y.buf, y.shape).astype(np.float32)
                     - tgt.numpy().astype(np.float32)) ** 2).mean())

        out = ref_forward(m, x)
        lt = float(F.mse_loss(out, tgt).detach())

        ex.backward(dy, y.shape)
        ex.step(1e-3, step, eps=1e-8 * scale)
        opt.zero_grad()
        F.mse_loss(ref_forward(m, x), tgt).backward()
        opt.step()

        rel = abs(ld - lt) / max(abs(lt), 1e-12)
        worst = max(worst, rel)
        print("  %4d  %9.6f  %9.6f  %9.2e" % (step, ld, lt, rel))
    print("RESULT " + ("PASS" if worst < 5e-2 else "FAIL"))
    return 0 if worst < 5e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
