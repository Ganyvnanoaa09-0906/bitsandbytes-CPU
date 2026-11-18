# -*- coding: utf-8 -*-
"""count_launches.py — 统计每步的 enqueue 次数（判定是否 launch-bound）。

计数方式：包住 Python 层的 kernel 访问（`_kernels()` 返回的 dict）与
`pyopencl.enqueue_copy`，不需要代理 CommandQueue（那会破坏 C 层类型检查）。

    python gt1030/count_launches.py --layers 4 --d 64
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import pyopencl as cl
import torch
import torch.nn as nn
import torch.nn.functional as F

import gt1030.igpu_fx as FX
import gt1030.igpu_graph as G
import gt1030.igpu_nn_ops as NN
import gt1030.igpu_ops2 as OPS2

SCALE = 65536.0
COUNT = {"kernel": 0, "copy": 0}


def _wrap_dict(d):
    class _D(dict):
        def __getitem__(self, name):
            fn = dict.__getitem__(self, name)

            def call(*a, **kw):
                COUNT["kernel"] += 1
                return fn(*a, **kw)
            return call
    return _D(d)


def install():
    for mod, attr in ((G, "_kernels"), (NN, "_kernels"), (OPS2, "kernels")):
        orig = getattr(mod, attr)
        setattr(mod, attr, lambda *a, _o=orig, **k: _wrap_dict(_o(*a, **k)))
    orig_copy = cl.enqueue_copy

    def copy(*a, **kw):
        COUNT["copy"] += 1
        return orig_copy(*a, **kw)
    cl.enqueue_copy = copy


class Blk(nn.Module):
    def __init__(self, d, nh, hd):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = FX.AttnModule(nh, hd)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, d * 2, bias=False)
        self.f2 = nn.Linear(d * 2, d, bias=False)
        with torch.no_grad():
            self.attn.o_proj.weight.mul_(0.5)
            self.f2.weight.mul_(0.5)

    def forward(self, x):
        h = x + self.attn(self.n1(x))
        return h + self.f2(F.gelu(self.f1(self.n2(h))))


class Stack(nn.Module):
    def __init__(self, layers, d, nh, hd, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Blk(d, nh, hd) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, tok):
        x = self.emb(tok)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=256)
    a = ap.parse_args()
    nh, hd = 4, a.d // 4

    torch.manual_seed(0)
    model = Stack(a.layers, a.d, nh, hd, a.vocab)
    toks = torch.randint(0, a.vocab, (a.T,))
    tgt = torch.randint(0, a.vocab, (a.T,))
    ex = FX.Executor(model, toks, state_bits=8, arena_mb=1200)
    ex.sync_from_torch()
    ex.set_schedule("sgdr", lr_max=1e-3, lr_min=1e-4, t0=5, t_mult=2, warmup=1)
    td = ex.dev.buf(4096)
    cl.enqueue_copy(ex.dev.q, td, np.ascontiguousarray(tgt.numpy(), dtype=np.int32))
    lb = ex.dev.buf(4096)
    gb = ex.dev.buf(max(4096, a.T * a.vocab * 2))

    def one_step(t):
        tr = torch.tensor
        ex.zero_grads()
        logits = ex.forward({"tok": toks.numpy()})
        OPS2.ce_forward(ex.dev, logits.buf, td, lb, gb, a.T, a.vocab, 1.0 / a.T, SCALE)
        ex.backward(gb, logits.shape)
        ex.step(None, t, eps=1e-8 * SCALE)
        ex.dev.q.finish()

    one_step(1)                     # 预热：编译 + 一次性分配
    install()
    COUNT["kernel"] = COUNT["copy"] = 0
    one_step(2)
    total = COUNT["kernel"] + COUNT["copy"]
    print("  %d 层 d=%d T=%d：每步 enqueue = %d（kernel %d + copy %d）"
          % (a.layers, a.d, a.T, total, COUNT["kernel"], COUNT["copy"]))
    print("  ⇒ 每层约 %.1f 次" % (total / a.layers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
