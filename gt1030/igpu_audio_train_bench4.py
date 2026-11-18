# -*- coding: utf-8 -*-
"""igpu_audio_train_bench4.py — 收尾验证：
  1) VITS 级（19M conv1d）在 bs=8/32 的 fwd+bwd，CPU vs iGPU；
  2) KWS 在 iGPU 上用 fp16 + 手动 loss scale，看能否收敛（DirectML 不支持 GradScaler 时的手动方案）。
"""
from __future__ import annotations

import json
import time

import torch
import torch.nn as nn

try:
    import torch_directml as dml
    HAS_DML = True
except Exception:
    HAS_DML = False

torch.set_num_threads(6)
OUT = r"D:\work\research\bench_results\audio_lab"


class VitsProxy(nn.Module):
    def __init__(self, ch=512, blocks=6):
        super().__init__()
        self.inp = nn.Conv1d(80, ch, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
                          nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU()) for _ in range(blocks)])
        self.out = nn.Conv1d(ch, 513, 1)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


class KWS(nn.Module):
    def __init__(self, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def bench_step(m, make, dev, n=5, warmup=2, dtype=torch.float32):
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    lf = nn.MSELoss()
    for _ in range(warmup):
        x, t = make(dev)
        opt.zero_grad()
        lf(m(x).float(), t.float()).backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(n):
        x, t = make(dev)
        opt.zero_grad()
        lf(m(x).float(), t.float()).backward()
        opt.step()
    return (time.perf_counter() - t0) / n


def main():
    res = {}
    devs = [("cpu", torch.device("cpu"))]
    if HAS_DML:
        devs.append(("igpu", dml.device(0)))

    for bs in (8, 32):
        for name, dev in devs:
            make = lambda d, bs=bs: (torch.randn(bs, 80, 100, device=d), torch.randn(bs, 513, 100, device=d))
            try:
                dt = bench_step(VitsProxy(), make, dev, n=3, warmup=1)
                res[f"vits_bs{bs}_{name}"] = round(dt, 3)
                print(f"  VITS bs={bs:<3} {name:<4} {dt*1000:7.0f} ms/步", flush=True)
            except Exception as e:
                res[f"vits_bs{bs}_{name}"] = dict(error=repr(e)[:120])
                print(f"  VITS bs={bs} {name} 失败: {repr(e)[:100]}", flush=True)

    # fp16 + 手动 loss scale
    if HAS_DML:
        try:
            dev = dml.device(0)
            torch.manual_seed(0)
            m = KWS().to(dev).half()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            lf = nn.CrossEntropyLoss()
            g = torch.Generator().manual_seed(0)
            X = torch.randn(256, 64, 64, generator=g).to(dev).half()
            y = torch.randint(0, 4, (256,), generator=g).to(dev)
            scale = 256.0
            t0 = time.perf_counter()
            for ep in range(20):
                perm = torch.randperm(256, device=dev)
                for i in range(0, 256, 64):
                    idx = perm[i:i + 64]
                    opt.zero_grad()
                    loss = lf(m(X[idx]).float(), y[idx]) * scale
                    loss.backward()
                    for p in m.parameters():
                        if p.grad is not None:
                            p.grad /= scale
                    opt.step()
            dt = time.perf_counter() - t0
            with torch.no_grad():
                acc = float((m(X).float().argmax(1) == y).float().mean())
            res["kws_fp16_scale_igpu"] = dict(seconds=round(dt, 2), acc=round(acc, 3))
            print(f"  KWS fp16+loss scale igpu 20 epoch: {dt:.2f}s | acc {acc:.3f}", flush=True)
        except Exception as e:
            res["kws_fp16_scale_igpu"] = dict(error=repr(e)[:150])
            print("  fp16+scale 失败:", repr(e)[:120], flush=True)

    with open(rf"{OUT}\igpu_train_bench4.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
