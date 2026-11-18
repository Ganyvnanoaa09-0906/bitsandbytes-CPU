# -*- coding: utf-8 -*-
"""ar_discriminator.py — 真图 token vs 模型生成 token 的判别器（"这画得像不像真的"的可测代理）。

为什么在 token 层做：VQ 解码到像素要过一遍 VQVAE，小差异会被量化抹掉（报告 §10 的结论）；
token 是逐位可比的，训练与推理都便宜（一个 32×32=1024 token 的序列，1.4M 参数模型）。

用途（两个）：
  1) 量化"生成 token 与真实 token 的可分性"（AUC 越低越好 = 越难分）；
  2) 当质量代理：拿它给不同模型（AR / MaskGIT）的样本打分，比较谁更接近真实分布。

用法：
  python ar_discriminator.py --real F:\\weibo_cos\\tokens_32x32_weibo.pt \
      --gen F:\\...\\samples.tokens.pt --steps 400 --out disc.pt
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


class Disc(nn.Module):
    def __init__(self, vocab=1024, side=32, d=192, layers=3, heads=4, ffn=768):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, side * side + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout=0.0, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, toks):
        h = self.emb(toks)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
        h = self.enc(h + self.pos[:, : h.shape[1]])
        return self.head(h[:, 0]).squeeze(-1)


def auc(scores_real, scores_fake):
    """真实样本得分应更高；返回 AUC（0.5=分不开，1.0=完美）."""
    r = np.asarray(scores_real, dtype=np.float64)
    f = np.asarray(scores_fake, dtype=np.float64)
    if len(r) == 0 or len(f) == 0:
        return float("nan")
    allv = np.concatenate([r, f])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 处理并列
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
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="")
    ap.add_argument("--score", nargs="*", default=[], help="额外打分的 tokens.pt（可多个）")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    real = torch.load(a.real, map_location="cpu", weights_only=False)
    if isinstance(real, dict):
        real = real.get("tokens", list(real.values())[0])
    real = real.long().view(real.shape[0], -1)
    gen = torch.load(a.gen, map_location="cpu", weights_only=False)
    gen = gen.long().view(gen.shape[0], -1)
    T = real.shape[1]
    side = int(round(T ** 0.5))
    print(f"  real {tuple(real.shape)} | gen {tuple(gen.shape)} | side={side}", flush=True)

    # 划分：real 的尾部 1024 张留作验证；gen 一半验证
    r_tr, r_va = real[:-1024], real[-1024:]
    g_va_n = max(1, gen.shape[0] // 2)
    g_tr, g_va = gen[: gen.shape[0] - g_va_n], gen[gen.shape[0] - g_va_n:]
    if g_va.shape[0] == 0:
        g_va = gen

    m = Disc(vocab=1024, side=side)
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"  判别器 {nparam/1e6:.2f}M 参数 | 训练 {tuple(r_tr.shape)} 真 / {tuple(g_tr.shape)} 假", flush=True)

    t0 = time.time()
    for step in range(a.steps):
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        ir = torch.randint(0, r_tr.shape[0], (a.batch,))
        ig = torch.randint(0, g_tr.shape[0], (a.batch,))
        x = torch.cat([r_tr[ir], g_tr[ig]], 0)
        y = torch.cat([torch.ones(a.batch), torch.zeros(a.batch)])
        opt.zero_grad()
        logits = m(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"    step {step+1}/{a.steps} loss={loss.item():.4f} lr={lr:.1e} "
                  f"{time.time()-t0:.0f}s", flush=True)

    m.eval()
    with torch.no_grad():
        sr = m(r_va).numpy()
        sg = m(g_va).numpy()
        auc_v = auc(sr, sg)
        print(f"  验证：真 {sr.mean():+.2f} vs 假 {sg.mean():+.2f} | **AUC {auc_v:.3f}**", flush=True)

    report = dict(params=nparam, auc=round(auc_v, 4), real_mean=float(sr.mean()),
                  gen_mean=float(sg.mean()), steps=a.steps, batch=a.batch, lr=a.lr,
                  real_n=int(real.shape[0]), gen_n=int(gen.shape[0]))
    # 额外打分
    extra = {}
    for p in a.score:
        t = torch.load(p, map_location="cpu", weights_only=False)
        t = t.long().view(t.shape[0], -1)
        with torch.no_grad():
            s = m(t).numpy()
        extra[p] = dict(mean=float(s.mean()), std=float(s.std()), n=int(t.shape[0]))
        print(f"  打分 {p}: mean {s.mean():+.2f} ± {s.std():.2f} (n={t.shape[0]}) "
              f"| 真 {sr.mean():+.2f}", flush=True)
    report["scores"] = extra
    if a.out:
        torch.save(m.state_dict(), a.out)
    with open(a.out + ".json" if a.out else "disc_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
