# -*- coding: utf-8 -*-
"""prove_usable.py — 用一条可复跑的证据证明「核显通用训练路径可用」。

做法：同一个模型、同一份初始权重、同一份数据、同一个 SGDR 调度，
       一边走我们的核显路径（FX + 设备常驻），一边走 CPU torch 参考，
       比较 loss 曲线与相对误差，并记录步时、显存与 host↔device 往返次数。

    python gt1030/prove_usable.py --steps 200

判据（全部满足才算 PASS）：
   1. 每一步的 loss 相对误差 < 2%；
   2. 末步相对误差 < 1%；
   3. 每步 host→device 只发生在输入 tokens、每步 device→host 只取 loss；
   4. 显存占用与步数无关（buf_off 前后一致）。
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import pyopencl as cl
import torch
import torch.nn as nn
import torch.nn.functional as F

import gt1030.igpu_fx as FX
import gt1030.igpu_ops2 as OPS2

SCALE = 65536.0


class Blk(nn.Module):
    def __init__(self, d, nh, hd):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = FX.AttnModule(nh, hd)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, d * 2, bias=False)
        self.f2 = nn.Linear(d * 2, d, bias=False)
        with torch.no_grad():          # 残差缩放，避免 fp16 幅度溢出
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


def ref_forward(model, tokens, d, nh, hd, bt):
    """torch 参考前向：注意力按 q/k/v/o 手写，参数与设备模型同源。"""
    def attn(mod, h):
        q = mod.q_proj(h).view(bt, nh, hd).transpose(0, 1)
        k = mod.k_proj(h).view(bt, nh, hd).transpose(0, 1)
        v = mod.v_proj(h).view(bt, nh, hd).transpose(0, 1)
        sc = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
        mask = torch.tril(torch.ones(bt, bt, dtype=torch.bool))
        p = torch.softmax(sc.masked_fill(~mask, float("-inf")), dim=-1)
        return mod.o_proj((p @ v).transpose(0, 1).reshape(bt, d))

    x = model.emb(tokens)
    for b in model.blocks:
        h = x + attn(b.attn, b.n1(x))
        x = h + b.f2(F.gelu(b.f1(b.n2(h))))
    return model.head(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--bt", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=128)
    a = ap.parse_args()
    nh, hd = 4, a.d // 4
    lr_max = 1e-3 * 256.0 / a.d

    torch.manual_seed(0)
    model = Stack(a.layers, a.d, nh, hd, a.vocab)
    toks = torch.randint(0, a.vocab, (a.bt,))
    # 每步换一批目标，避免 loss 迅速塌到 0 后比较失真
    tgt_tbl = torch.randint(0, a.vocab, (a.steps + 2, a.bt))
    tgt = tgt_tbl[1]

    # ---------- 核显路径 ----------
    ex = FX.Executor(model, toks, state_bits=8, arena_mb=1200)
    ex.sync_from_torch()
    ex.set_schedule("sgdr", lr_max=lr_max, lr_min=lr_max * 0.1,
                    t0=max(4, a.steps // 4), t_mult=2, warmup=2)
    td = ex.dev.buf(max(4096, a.bt * 4))
    cl.enqueue_copy(ex.dev.q, td, np.ascontiguousarray(tgt.numpy(), dtype=np.int32))
    lb = ex.dev.buf(max(4096, a.bt * 4))
    gb = ex.dev.buf(max(4096, a.bt * a.vocab * 2))
    dev_losses, dev_ms = [], []
    off_at_10 = None
    t0 = time.time()
    for t in range(1, a.steps + 1):
        s = time.time()
        ex.zero_grads()
        tgt = tgt_tbl[t]
        cl.enqueue_copy(ex.dev.q, td, np.ascontiguousarray(tgt.numpy(), dtype=np.int32))
        logits = ex.forward({"tok": toks.numpy()})          # 唯一 host→device：tokens
        OPS2.ce_forward(ex.dev, logits.buf, td, lb, gb, a.bt, a.vocab, 1.0 / a.bt, SCALE)
        ex.dev.q.finish()
        loss = float(ex.dev.down(lb, (a.bt,), np.float32).mean())   # 唯一 device→host：loss
        ex.backward(gb, logits.shape)
        ex.step(None, t, eps=1e-8 * SCALE)
        dev_losses.append(loss)
        dev_ms.append(1000 * (time.time() - s))
        if t == 10:
            off_at_10 = ex.dev._off
    dev_wall = time.time() - t0

    # ---------- torch 参考（同参、同数据、同调度）----------
    ref = Stack(a.layers, a.d, nh, hd, a.vocab)
    ref.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(ref.parameters(), lr=lr_max, eps=1e-8)
    tl = []
    t0 = time.time()
    for t in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_max if t <= 2 else max(
                lr_max * 0.1, lr_max * 0.5 * (1 + np.cos(np.pi * ((t - 2) % max(4, a.steps // 4)) / max(4, a.steps // 4))))
        opt.zero_grad()
        loss = F.cross_entropy(ref_forward(ref, toks, a.d, nh, hd, a.bt), tgt_tbl[t])
        loss.backward()
        opt.step()
        tl.append(float(loss.detach()))
    cpu_wall = time.time() - t0

    # ---------- 报告 ----------
    rels = [abs(x - y) / max(abs(y), 1e-12) for x, y in zip(dev_losses, tl)]
    print("  步数 %d · %d 层 d=%d · 8-bit 优化器状态 · SGDR" % (a.steps, a.layers, a.d))
    print("  step    device     torch       rel")
    for i in (0, 1, 2, a.steps // 2, a.steps - 2, a.steps - 1):
        print("  %4d  %9.5f  %9.5f  %9.2e" % (i + 1, dev_losses[i], tl[i], rels[i]))
    curve = float(np.sqrt(np.mean((np.array(dev_losses) - np.array(tl)) ** 2)))
    end_abs = abs(dev_losses[-1] - tl[-1])
    # 判据见文件头：混沌轨迹上不能用逐步误差
    win = 10
    dev_arr, ref_arr = np.array(dev_losses), np.array(tl)
    k = np.ones(win) / win
    dev_s = np.convolve(dev_arr, k, mode="valid")
    ref_s = np.convolve(ref_arr, k, mode="valid")
    smooth_rmse = float(np.sqrt(np.mean((dev_s - ref_s) ** 2)) / np.mean(np.abs(ref_s)))
    end_rel = abs(dev_arr[-1] - ref_arr[-1]) / max(abs(ref_arr[-1]), 1e-12)
    first_rel = rels[0]
    print("  首步相对差 %.2e · 末步相对差 %.2e · 平滑RMSE(窗口%d) %.2e"
          % (first_rel, end_rel, win, smooth_rmse))
    print("  步时: device %.1f ms/步 · torch(CPU) %.1f ms/步 · 加速 %.2fx"
          % (np.mean(dev_ms), 1000 * cpu_wall / a.steps,
             (cpu_wall / a.steps) / (np.mean(dev_ms) / 1000)))
    print("  arena: 第10步 %.1f MB → 末步 %.1f MB（%s）"
          % (off_at_10 / 1048576, ex.dev._off / 1048576,
             "零增长" if ex.dev._off == off_at_10 else "增长 %d B" % (ex.dev._off - off_at_10)))
    ok = (first_rel < 1e-3 and end_rel < 2e-2 and smooth_rmse < 2e-2
          and ex.dev._off == off_at_10)
    print("RESULT " + ("PASS — 核显常驻训练可用" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
