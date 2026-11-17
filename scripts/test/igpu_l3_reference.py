# -*- coding: utf-8 -*-
"""igpu_l3_reference.py — L3 验收参考：**N 层 transformer stack** 的 CPU fp32 训练。

与 L1 同一套用法（`--out ref.json` / `--compare device.json`），但模型是完整 stack：
    x[B,T,d]
    └ Block ×N:  RMSNorm → causal MHA(含 QKV/out 投影) → +x → RMSNorm → SwiGLU MLP → +x
    loss = MSE(y, target)

默认（与 bluearmy 确认过的规格）：
    N=4, d=512, n_head=8, head_dim=64, ffn=1368, B=2, T=256, steps=20, lr=1e-5, seed=0

★ device 侧请**直接 import 本文件的 `build_model` / `make_data`** 来获得同一份初值与数据
  （torch 顺序建模型即可，随后一次性上传；不要自己另写 init，否则不可比）。

用法：
    python scripts/test/igpu_l3_reference.py --out ref_l3.json
    python scripts/test/igpu_l3_reference.py --out ref_l3.json --compare device_l3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = dict(n_layers=4, d=512, nh=8, hd=64, ffn=1368, b=2, t=256,
                steps=20, lr=1e-5, seed=0, eps=1e-6)


# ---------------------------------------------------------------- 组件（语义与 device 侧一致）
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rs = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rs * self.w


class MHA(nn.Module):
    """causal multi-head attention：QKV 融合投影 + out 投影，scale = 1/sqrt(hd)。"""

    def __init__(self, d, nh, hd):
        super().__init__()
        self.nh, self.hd, self.d = nh, hd, d
        self.qkv = nn.Linear(d, 3 * nh * hd, bias=False)
        self.out = nn.Linear(nh * hd, d, bias=False)
        self.scale = 1.0 / math.sqrt(hd)

    def forward(self, x):                       # x: (B, T, d)
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]        # (B, nh, T, hd)
        s = torch.einsum("bhtd,bhud->bhtu", q, k) * self.scale
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("bhtu,bhud->bhtd", p, v)          # (B, nh, T, hd)
        o = o.permute(0, 2, 1, 3).reshape(B, T, self.nh * self.hd)
        return self.out(o)


class SwiGLU(nn.Module):
    """SwiGLU：w12 输出 2F，g/u 拆半，y = w3(silu(g) * u)。"""

    def __init__(self, d, ffn):
        super().__init__()
        self.w12 = nn.Linear(d, 2 * ffn, bias=False)
        self.w3 = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        a = self.w12(x)
        g, u = a.chunk(2, dim=-1)
        return self.w3(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, d, nh, hd, ffn, eps=1e-6):
        super().__init__()
        self.n1 = RMSNorm(d, eps)
        self.attn = MHA(d, nh, hd)
        self.n2 = RMSNorm(d, eps)
        self.mlp = SwiGLU(d, ffn)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class Stack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg["d"], cfg["nh"], cfg["hd"], cfg["ffn"], cfg["eps"])
                                     for _ in range(cfg["n_layers"])])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def build_model(cfg, seed):
    torch.manual_seed(seed)
    return Stack(cfg)


def make_data(cfg, seed):
    g = torch.Generator().manual_seed(seed + 12345)
    x = torch.randn(cfg["b"], cfg["t"], cfg["d"], generator=g) * 0.02
    target = torch.randn(cfg["b"], cfg["t"], cfg["d"], generator=g) * 0.02
    return x, target


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()[:16]


def run(cfg):
    model = build_model(cfg, cfg["seed"])
    x, target = make_data(cfg, cfg["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0)
    losses, gn, step_times = [], {}, []
    for step in range(cfg["steps"]):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        if step == 0:
            gn = {
                "attn_qkv": float(torch.cat([b.attn.qkv.weight.grad.norm().reshape(1)
                                             for b in model.blocks]).norm()),
                "attn_out": float(torch.cat([b.attn.out.weight.grad.norm().reshape(1)
                                             for b in model.blocks]).norm()),
                "mlp_w12": float(torch.cat([b.mlp.w12.weight.grad.norm().reshape(1)
                                            for b in model.blocks]).norm()),
                "mlp_w3": float(torch.cat([b.mlp.w3.weight.grad.norm().reshape(1)
                                           for b in model.blocks]).norm()),
            }
        opt.step()
        losses.append(float(loss.detach()))
        step_times.append(time.perf_counter() - t0)
    weights = {"blk0_attn_qkv": tensor_sha(model.blocks[0].attn.qkv.weight),
               "blk0_mlp_w12": tensor_sha(model.blocks[0].mlp.w12.weight),
               "blk_last_out": tensor_sha(model.blocks[-1].attn.out.weight)}
    return {
        "config": dict(cfg),
        "dtype": "fp32-cpu",
        "losses": losses,
        "final_loss": losses[-1],
        "first_grad_norms": gn,
        "weights_sha256_16": weights,
        "step_seconds_median": statistics.median(step_times),
        "params": sum(p.numel() for p in model.parameters()),
    }


def compare(ref, dev, loss_rtol, final_rtol):
    ok = True
    if len(dev.get("losses", [])) != len(ref["losses"]):
        print("  FAIL: loss count mismatch: ref %d vs dev %d"
              % (len(ref["losses"]), len(dev.get("losses", []))))
        return False
    worst = 0.0
    for i, (a, b) in enumerate(zip(ref["losses"], dev["losses"])):
        if b != b or b in (float("inf"), float("-inf")):
            print("  FAIL: non-finite loss at step %d" % i)
            return False
        rel = abs(a - b) / max(abs(a), 1e-12)
        worst = max(worst, rel)
        if rel > loss_rtol:
            print("  FAIL: step %d loss %.6g vs ref %.6g (rel %.3f > %.3f)"
                  % (i, b, a, rel, loss_rtol))
            ok = False
    frel = abs(dev["final_loss"] - ref["final_loss"]) / max(abs(ref["final_loss"]), 1e-12)
    print("  worst per-step rel diff: %.3f | final rel diff: %.3f" % (worst, frel))
    if frel > final_rtol:
        print("  FAIL: final loss outside %.3f" % final_rtol)
        ok = False
    if "peak_device_mb" in dev and dev["peak_device_mb"] > 4850:
        print("  FAIL: peak_device_mb %.0f > 4850" % dev["peak_device_mb"])
        ok = False
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ty = float if k in ("eps", "lr") else int
        ap.add_argument("--" + k.replace("_", "-"), type=ty, default=v)
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("--loss-rtol", type=float, default=0.05)
    ap.add_argument("--final-rtol", type=float, default=0.05)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    if a.compare:
        ref = json.load(open(a.out, encoding="utf-8")) if a.out else run(cfg)
        dev = json.load(open(a.compare, encoding="utf-8"))
        sys.exit(0 if compare(ref, dev, a.loss_rtol, a.final_rtol) else 1)
    res = run(cfg)
    print("  params %.2fM | losses %.6g -> %.6g | step %.1f ms | grads %s"
          % (res["params"] / 1e6, res["losses"][0], res["final_loss"],
             res["step_seconds_median"] * 1e3,
             {k: round(v, 6) for k, v in res["first_grad_norms"].items()}))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
        print("  wrote", a.out)


if __name__ == "__main__":
    main()
