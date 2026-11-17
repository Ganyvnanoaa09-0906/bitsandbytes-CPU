# -*- coding: utf-8 -*-
"""probe_stack.py — 逐节点对比 device 与 torch 中间量，定位多层注意力从哪一步分叉。"""
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gt1030.igpu_fx as FX

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


def torch_attn(mod, h):
    q = mod.q_proj(h).view(bt, nh, hd).transpose(0, 1)
    k = mod.k_proj(h).view(bt, nh, hd).transpose(0, 1)
    v = mod.v_proj(h).view(bt, nh, hd).transpose(0, 1)
    s = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
    mask = torch.tril(torch.ones(bt, bt, dtype=torch.bool))
    p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
    o = (p @ v).transpose(0, 1).reshape(bt, d)
    return mod.o_proj(o)


def main() -> int:
    torch.manual_seed(0)
    m = S()
    x = torch.randn(bt, d) * 0.3
    ex = FX.Executor(m, x)
    ex.sync_from_torch()
    ex.forward(x.numpy())

    t = {}
    cur = x
    add_names = ["add", "add_2"]
    out_names = ["add_1", "add_3"]
    for i, blk in enumerate(m.blocks):
        n1 = blk.n1(cur)
        a = torch_attn(blk.attn, n1)
        h = cur + a
        n2 = blk.n2(h)
        f1o = F.gelu(blk.f1(n2))
        f2o = blk.f2(f1o)
        cur = h + f2o
        t["blocks_%d_n1" % i] = n1
        t["blocks_%d_attn" % i] = a
        t[add_names[i]] = h
        t["blocks_%d_f2" % i] = f2o
        t[out_names[i]] = cur

    order = ["blocks_0_n1", "blocks_0_attn", "add", "blocks_0_f2", "add_1",
             "blocks_1_n1", "blocks_1_attn", "add_2", "blocks_1_f2", "add_3"]
    print("  node              device     torch        rel")
    for name in order:
        obj = ex.env.get(name)
        if obj is None:
            print("  %-16s <no device node>" % name)
            continue
        v = ex.dev.down(obj.buf, obj.shape).astype(np.float32)
        r = t[name].detach().numpy()
        rel = float(np.linalg.norm(v - r) / (np.linalg.norm(r) + 1e-12))
        print("  %-16s %10.4f %10.4f %10.2e" % (name, np.linalg.norm(v), np.linalg.norm(r), rel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
