# -*- coding: utf-8 -*-
"""probe_sumsq.py — 逐层对比：kernel 算出的平方和 vs 把同一 buffer 下载到 host 用 fp64 算的平方和。

如果两者相差很大，说明 grad_norm 的归约 kernel 有问题（而不是梯度本身有问题）。
"""
import os
import sys

import numpy as np

REPO = r"D:\work\bitsandbytes-CPU"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "gt1030"))
os.environ.setdefault("AR_ARENA_MB", "3200")
import ar_v2_multi as MM  # noqa: E402

TOK = r"D:\work\cloud_salvage\ScPeP7\tokens_V32.pt"
CKPT = sys.argv[1] if len(sys.argv) > 1 else r"F:\weibo_cos\ar_cpu2\ckpt_46000.pt"

import torch
tk = torch.load(TOK, map_location="cpu", weights_only=False)
tk = (tk.get("tokens") if isinstance(tk, dict) else tk).long()
rng = np.random.RandomState(0)
idx = rng.choice(len(tk), 2, replace=False)
tokens = np.ascontiguousarray(tk[idx].numpy()[:, :MM.T])

m = MM.Model(CKPT)
loss = m.forward(tokens)
m.backward(tokens)
print(f"[probe] loss={loss:.6f}  Model.grad_norm()={m.grad_norm():.4f}", flush=True)

print(f"\n{'layer':>5} {'name':>5} {'kernel':>14} {'host fp64':>14} {'比值':>8}")
tot_k = tot_h = 0.0
for bi, blk in enumerate(m.blocks):
    d = blk.dev
    d.activate()
    for k in ("Wqkv", "Wo", "W12", "W3"):
        p = blk.W[k]
        ker = d.sumsq_many([(p.grad, p.n)])[0]
        g = d.down(p.grad, (p.n,), np.float16).astype(np.float64)
        host = float(np.sum(g * g))
        tot_k += ker
        tot_h += host
        print(f"{bi:>5} {k:>5} {ker:>14.4f} {host:>14.4f} {ker/max(host,1e-30):>8.2e}")
h = m.head
h.dev.activate()
ker = h.dev.sumsq_many([(h.W.grad, h.W.n)])[0]
gh = h.dev.down(h.W.grad, (h.W.n,), np.float16).astype(np.float64)
host = float(np.sum(gh * gh))
tot_k += ker
tot_h += host
print(f"{'head':>5} {'W':>5} {ker:>14.4f} {host:>14.4f} {ker/max(host,1e-30):>8.2e}")
print(f"\n合计 kernel={tot_k:.4f}（范数 {np.sqrt(tot_k):.4f}）  host={tot_h:.4f}（范数 {np.sqrt(tot_h):.4f}）")
print(f"比值 kernel/host = {tot_k/max(tot_h,1e-30):.3e}")
