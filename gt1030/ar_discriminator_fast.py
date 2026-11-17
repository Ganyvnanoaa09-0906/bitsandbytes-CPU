# -*- coding: utf-8 -*-
"""ar_discriminator_fast.py — 轻量版判别器（卷积下采样，比 token-transformer 快 ~10×）。

结构：token embedding(1024→d) → 2D 网格 → 两层 stride-2 卷积 → 全局平均池化 → MLP → 1 logit。
32×32 grid：32 → 16 → 8，再池化，计算量很小（1.7M 参数，但每步只做卷积）。

用途：量化"生成 token 与真实 token 的可分性"（AUC），并给不同模型的样本打分（质量代理）。

用法：
  python ar_discriminator_fast.py --real F:\\weibo_cos\\tokens_32x32_weibo.pt \
      --gen F:\\weibo_cos\\_ar76400_samples.tokens.pt \
      --score F:\\weibo_cos\\_mgit_b4\\samples.tokens.pt F:\\weibo_cos\\_mgit_b0\\samples.tokens.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscFast(nn.Module):
    def __init__(self, vocab=1024, side=32, d=128):
        super().__init__()
        self.side = side
        self.emb = nn.Embedding(vocab, d)
        self.c1 = nn.Conv2d(d, 2 * d, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(2 * d, 4 * d, 3, stride=2, padding=1)
        self.n1 = nn.GroupNorm(8, 2 * d)
        self.n2 = nn.GroupNorm(8, 4 * d)
        self.head = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, toks):
        B = toks.shape[0]
        h = self.emb(toks).reshape(B, self.side, self.side, -1).permute(0, 3, 1, 2)
        h = F.gelu(self.n1(self.c1(h)))
        h = F.gelu(self.n2(self.c2(h)))
        h = h.mean(dim=(2, 3))
        return self.head(h).squeeze(-1)


def auc(scores_real, scores_fake):
    r = np.asarray(scores_real, dtype=np.float64)
    f = np.asarray(scores_fake, dtype=np.float64)
    if len(r) == 0 or len(f) == 0:
        return float("nan")
    allv = np.concatenate([r, f])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    for gi in range(len(counts)):
        if counts[gi] > 1:
            m = inv == gi
            ranks[m] = ranks[m].mean()
    rsum = ranks[: len(r)].sum()
    return float((rsum - len(r) * (len(r) + 1) / 2) / (len(r) * len(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--score", nargs="*", default=[])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    def load(p):
        t = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            t = t.get("tokens", list(t.values())[0])
        return t.long().view(t.shape[0], -1)

    real = load(a.real)
    gen = load(a.gen)
    T = real.shape[1]
    side = int(round(T ** 0.5))
    print(f"  real {tuple(real.shape)} | gen {tuple(gen.shape)} | side={side}", flush=True)

    r_tr, r_va = real[:-1024], real[-1024:]
    n_va = max(1, gen.shape[0] // 2)
    g_tr, g_va = gen[: gen.shape[0] - n_va], gen[gen.shape[0] - n_va:]

    m = DiscFast(vocab=1024, side=side)
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"  判别器(conv) {nparam/1e6:.2f}M 参数 | 真 {r_tr.shape[0]} / 假 {g_tr.shape[0]}", flush=True)

    t0 = time.time()
    for step in range(a.steps):
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        ir = torch.randint(0, r_tr.shape[0], (a.batch,))
        ig = torch.randint(0, max(1, g_tr.shape[0]), (a.batch,))
        x = torch.cat([r_tr[ir], g_tr[ig % max(1, g_tr.shape[0])]], 0)
        y = torch.cat([torch.ones(a.batch), torch.zeros(a.batch)])
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(m(x), y)
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == a.steps - 1:
            print(f"    step {step+1}/{a.steps} loss={loss.item():.4f} {time.time()-t0:.0f}s", flush=True)

    m.eval()
    with torch.no_grad():
        sr = m(r_va).numpy()
        sg = m(g_va).numpy()
        auc_v = auc(sr, sg)
    print(f"  验证：真 {sr.mean():+.2f} vs 假 {sg.mean():+.2f} | AUC {auc_v:.3f}", flush=True)

    rep = dict(params=nparam, auc=round(auc_v, 4), real_mean=float(sr.mean()),
               gen_mean=float(sg.mean()), steps=a.steps, batch=a.batch,
               real_n=int(real.shape[0]), gen_n=int(gen.shape[0]))
    extra = {}
    for p in a.score:
        t = load(p)
        with torch.no_grad():
            s = m(t).numpy()
        extra[p] = dict(mean=float(s.mean()), std=float(s.std()), n=int(t.shape[0]))
        print(f"  打分 {p.split(chr(92))[-1]}: mean {s.mean():+.2f} ± {s.std():.2f} (n={t.shape[0]})", flush=True)
    rep["scores"] = extra
    if a.out:
        torch.save(m.state_dict(), a.out)
        with open(a.out + ".json", "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
