# -*- coding: utf-8 -*-
"""make_maskgit_init.py — 把 1024 词表的 AR ckpt 扩成 1025（多一个 MASK id），用于 MaskGIT 训练。

三个 (1024, 512) 张量（token_emb / gen_head / think_head）各追加一行：复制末行作为 MASK 的初值
（与报告里早先的做法一致），cfg.vocab_size 改成 1025。

用法：
  python make_maskgit_init.py --src F:\\weibo_cos\\_from76000\\ckpt_76400.pt \
                              --dst F:\\weibo_cos\\_mgit_init\\ckpt_76400_mg.pt
"""
from __future__ import annotations

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.dst), exist_ok=True)
    d = torch.load(a.src, map_location="cpu", weights_only=False)
    sd = d["model"]
    old_v = int(d["cfg"]["vocab_size"])
    new_v = old_v + 1
    n = 0
    for k in ("token_emb.weight", "gen_head.lin.weight", "think_head.lin.weight"):
        if k not in sd:
            print(f"  [skip] 缺少 {k}")
            continue
        w = sd[k]
        if w.shape[0] != old_v:
            print(f"  [skip] {k} 形状 {tuple(w.shape)} 不是词表维")
            continue
        sd[k] = torch.cat([w, w[-1:].clone()], dim=0)
        n += 1
    d["cfg"]["vocab_size"] = new_v
    d["step"] = int(d.get("step", 0))
    torch.save(d, a.dst)
    print(f"  [ok] 扩展 {n} 个张量：{old_v} → {new_v}；MASK id = {new_v - 1}")
    print(f"  保存 {a.dst}（{os.path.getsize(a.dst)/1e6:.0f} MB）| step={d['step']}")


if __name__ == "__main__":
    main()
