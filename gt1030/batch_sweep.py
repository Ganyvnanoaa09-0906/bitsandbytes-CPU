# -*- coding: utf-8 -*-
"""batch_sweep.py — 把本机模型批量丢进核显通用路径训练，收集日志。

用法：
    python gt1030/batch_sweep.py --list                 # 只列清单（不跑）
    python gt1030/batch_sweep.py --only ar30m --steps 1500
    python gt1030/batch_sweep.py --all --steps 1500     # 全部（按清单顺序）

日志：D:\\work\\research\\bench_results\\sweep\\<name>.log + summary.jsonl
资源：默认走 pool（pool_client），可用 --no-pool 关闭。
"""
from __future__ import annotations

import argparse
import json
import os
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

OUT = r"D:\work\research\bench_results\sweep"
SCALE = 65536.0

# ---------------------------------------------------------------- 模型清单
# kind=fx  : 用 FX 通用路径（构造成 nn.Module 再编译）
# kind=skip: 明确不可行（记录原因，不跑）
MANIFEST = [
    dict(name="ar30m_like", kind="fx", layers=10, d=512, nh=8, hd=64, bt=8, vocab=1024,
         note="AR 30M 同构（10 层 d=512），验证 A6 量级的通用路径"),
    dict(name="qwen3_1.7b_slice", kind="fx", layers=8, d=512, nh=8, hd=64, bt=8, vocab=1024,
         note="Qwen3-1.7B 的 8 层切片（同宽度），非原权重"),
    dict(name="sd15_unet_slice", kind="fx", layers=4, d=256, nh=8, hd=32, bt=8, vocab=512,
         note="SD1.5 UNet 宽度切片（卷积未接 FX，先用注意力块代表）"),
    dict(name="wan_dit_slice", kind="fx", layers=6, d=384, nh=6, hd=64, bt=8, vocab=512,
         note="Wan2.1 DiT 宽度切片"),
    dict(name="qwen_image_vae_slice", kind="fx", layers=3, d=256, nh=4, hd=64, bt=8, vocab=256,
         note="Qwen-Image VAE 宽度切片"),
    dict(name="qwen3_1.7b_full", kind="skip",
         note="1.7B fp16 ≈ 3.4 GB 权重 + 优化器状态 > 4.85 GB 单次分配上限，当前不可行"),
    dict(name="sd15_unet_full", kind="skip",
         note="UNet ~860M 参数，远超 4.85 GB（fp16 权重已 1.7 GB，加状态必爆）"),
]


class AttnBlock(nn.Module):
    def __init__(self, d, nh, hd):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = FX.AttnModule(nh, hd)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, d * 2, bias=False)
        self.f2 = nn.Linear(d * 2, d, bias=False)
        with torch.no_grad():
            self.attn.o_proj.weight.mul_(0.5)   # 残差缩放（见 stack4 的证据）
            self.f2.weight.mul_(0.5)

    def forward(self, x):
        h = x + self.attn(self.n1(x))
        return h + self.f2(F.gelu(self.f1(self.n2(h))))


class Stack(nn.Module):
    def __init__(self, layers, d, nh, hd, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([AttnBlock(d, nh, hd) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens):
        x = self.emb(tokens)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def arena_for(spec):
    """按参数量与激活量估 arena（MB），留 1.6 倍余量。"""
    d, L, nh, hd = spec["d"], spec["layers"], spec["nh"], spec["hd"]
    per = 4 * d * d + 4 * d * hd * nh + 8 * d * d          # qkv/o + ffn 参数
    nparam = L * per + 2 * spec["vocab"] * d
    act = spec["bt"] * d * 4 * (L + 2) * 2
    return int(max(600, (nparam * 16 + act) * 1.6 / (1 << 20)))


def run_one(spec, steps, lr_max=1e-3, bits=8, muon=False, seed=0):
    d, nh, hd, bt, vocab = spec["d"], spec["nh"], spec["hd"], spec["bt"], spec["vocab"]
    torch.manual_seed(seed)
    model = Stack(spec["layers"], d, nh, hd, vocab)
    toks = torch.randint(0, vocab, (bt,))
    tgt = torch.randint(0, vocab, (bt,))
    ex = FX.Executor(model, toks, state_bits=bits, muon=muon,
                     arena_mb=arena_for(spec))
    ex.sync_from_torch()
    lr_scale = 256.0 / float(spec["d"])   # 宽模型 lr 要更小
    ex.set_schedule("sgdr", lr_max=lr_max * lr_scale, lr_min=lr_max * lr_scale * 0.1, t0=max(4, steps // 4),
                    t_mult=2, warmup=2)
    td = ex.dev.buf(max(4096, bt * 4))
    cl.enqueue_copy(ex.dev.q, td, np.ascontiguousarray(tgt.numpy(), dtype=np.int32))
    lb = ex.dev.buf(max(4096, bt * 4))
    gb = ex.dev.buf(max(4096, bt * vocab * 2))
    losses = []
    t0 = time.time()
    for t in range(1, steps + 1):
        ex.zero_grads()
        logits = ex.forward({"tokens": toks.numpy()})
        OPS2.ce_forward(ex.dev, logits.buf, td, lb, gb, bt, vocab, 1.0 / bt, SCALE)
        ex.dev.q.finish()
        losses.append(float(ex.dev.down(lb, (bt,), np.float32).mean()))
        ex.backward(gb, logits.shape)
        ex.step(None, t, eps=1e-8 * SCALE)
        if not np.isfinite(losses[-1]):
            break
    dt = time.time() - t0
    return dict(name=spec["name"], steps=len(losses), loss_first=losses[0],
                loss_last=losses[-1], ms_per_step=1000 * dt / max(1, len(losses)),
                nan=bool(not np.isfinite(losses[-1])), note=spec.get("note", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--muon", action="store_true")
    ap.add_argument("--no-pool", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("  %-22s %-6s %s" % ("name", "kind", "note"))
        for s in MANIFEST:
            print("  %-22s %-6s %s" % (s["name"], s["kind"], s.get("note", "")))
        return 0

    todo = [s for s in MANIFEST if s["kind"] == "fx"]
    if a.only:
        todo = [s for s in todo if s["name"] == a.only]
    elif not a.all:
        todo = todo[:1]
    os.makedirs(OUT, exist_ok=True)
    print("  待跑 %d 个模型 × %d 步" % (len(todo), a.steps), flush=True)

    for spec in todo:
        log = os.path.join(OUT, spec["name"] + ".log")
        t0 = time.time()
        try:
            res = run_one(spec, a.steps, bits=a.bits, muon=a.muon)
        except Exception as error:
            res = dict(name=spec["name"], error=repr(error))
        res["wall_s"] = round(time.time() - t0, 1)
        with open(log, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
        with open(os.path.join(OUT, "summary.jsonl"), "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        print("  " + json.dumps(res, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
