# -*- coding: utf-8 -*-
"""stack4_matrix.py — 4 层 transformer stack 的优化器/状态验收矩阵（SGDR，20 步）。

    python gt1030/stack4_matrix.py
"""
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import pyopencl as cl
import torch
import torch.nn as nn
import torch.nn.functional as F

import gt1030.igpu_fx as FX
import gt1030.igpu_graph as G
import gt1030.igpu_ops2 as OPS2

d, nh, hd, bt, vocab = 32, 4, 8, 16, 64
SCALE = 65536.0


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = FX.AttnModule(nh, hd)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, d * 2, bias=False)
        self.f2 = nn.Linear(d * 2, d, bias=False)
        # 深层残差缩放：把两条残差支路的输出投影各乘 0.5（等价于 0.5*block(x)），
        # 避免 fp16 激活幅度随层数累加溢出 65504。直接改初始权重，不引入新的算子。
        with torch.no_grad():
            self.attn.o_proj.weight.mul_(0.5)
            self.f2.weight.mul_(0.5)

    def forward(self, x):
        h = x + self.attn(self.n1(x))
        return h + self.f2(F.gelu(self.f1(self.n2(h))))


class Stack(nn.Module):
    def __init__(self, layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block() for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens):
        x = self.emb(tokens)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def run(layers=4, bits=32, muon=False, muon_lr=0.005, steps=20, lr_max=1e-3):
    torch.manual_seed(0)
    model = Stack(layers)
    toks = torch.randint(0, vocab, (bt,))
    tgt = torch.randint(0, vocab, (bt,))
    ex = FX.Executor(model, toks, state_bits=bits, muon=muon, muon_lr=muon_lr)
    ex.sync_from_torch()
    ex.set_schedule("sgdr", lr_max=lr_max, lr_min=lr_max * 0.1, t0=8, t_mult=2, warmup=2)
    td = ex.dev.buf(max(4096, bt * 4))
    cl.enqueue_copy(ex.dev.q, td, np.ascontiguousarray(tgt.numpy(), dtype=np.int32))
    lb = ex.dev.buf(max(4096, bt * 4))
    gb = ex.dev.buf(max(4096, bt * vocab * 2))
    losses, t0 = [], time.time()
    for t in range(1, steps + 1):
        ex.zero_grads()
        logits = ex.forward({"tokens": toks.numpy()})
        OPS2.ce_forward(ex.dev, logits.buf, td, lb, gb, bt, vocab, 1.0 / bt, SCALE)
        ex.dev.q.finish()
        losses.append(float(ex.dev.down(lb, (bt,), np.float32).mean()))
        ex.backward(gb, logits.shape)
        ex.step(None, t, eps=1e-8 * SCALE)
    dt = time.time() - t0
    return losses, dt


def main() -> int:
    cells = [
        ("AdamW fp32", dict(bits=32)),
        ("AdamW 8bit", dict(bits=8)),
        ("AdamW 4bit", dict(bits=4)),
        ("Muon+fp32", dict(bits=32, muon=True)),
        ("Muon+8bit", dict(bits=8, muon=True)),
        ("Muon+4bit", dict(bits=4, muon=True)),
    ]
    print("  4 层 stack · SGDR · 20 步")
    print("  %-12s %10s %10s %10s %9s" % ("配置", "loss@1", "loss@20", "Δ", "ms/步"))
    ok = True
    for name, kw in cells:
        ls, dt = run(**kw)
        bad = any(np.isnan(ls))
        ok &= not bad
        print("  %-12s %10.6f %10.6f %10.6f %9.1f %s"
              % (name, ls[0], ls[-1], ls[-1] - ls[0], 1000 * dt / len(ls),
                 "NaN" if bad else ""))
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
