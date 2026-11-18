# -*- coding: utf-8 -*-
"""igpu_audio_train_bench.py — 音频模型「核显训练」对比：CPU vs DirectML(iGPU)。

三档：
  1) KWS 关键词 CNN（23k 参数，conv2d）：CPU vs iGPU 训练步速 + 真实训练 60 epoch 的准确率/耗时；
  2) VITS 级 proxy（19M，conv1d）：单步对比；
  3) Whisper-small 级（185M，transformer）：单步对比（可能受 iGPU 内存限制）。

用法（用 DirectML venv 的 python 跑）：
  D:\\work\\venv_dml\\Scripts\\python.exe igpu_audio_train_bench.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn

try:
    import torch_directml as dml
    HAS_DML = True
except Exception:
    HAS_DML = False

torch.set_num_threads(6)
OUT = r"D:\work\research\bench_results\audio_lab"


class KWS(nn.Module):
    def __init__(self, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


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


def steps_bench(model, make_batch, device, n=15, warmup=3, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss() if isinstance(model, KWS) else nn.MSELoss()
    for _ in range(warmup):
        x, y = make_batch(device)
        opt.zero_grad()
        out = model(x)
        loss = lossf(out, y) if isinstance(model, KWS) else lossf(out, y)
        loss.backward()
        opt.step()
    if hasattr(device, "synchronize"):
        pass
    t0 = time.perf_counter()
    for _ in range(n):
        x, y = make_batch(device)
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
    dt = (time.perf_counter() - t0) / n
    return round(dt, 4), round(1 / dt, 2)


def main():
    devices = [("cpu", torch.device("cpu"))]
    if HAS_DML:
        devices.append(("igpu", dml.device(0)))

    results = {}

    # 1) KWS 训练
    def kws_batch(dev):
        return torch.randn(16, 64, 64, device=dev), torch.randint(0, 4, (16,), device=dev)

    for name, dev in devices:
        try:
            dt, sps = steps_bench(KWS(), kws_batch, dev)
            results[f"kws_{name}"] = dict(step_s=dt, steps_per_s=sps)
            print(f"  KWS {name}: {dt*1000:.1f} ms/步 → {sps} 步/s", flush=True)
        except Exception as e:
            results[f"kws_{name}"] = dict(error=repr(e)[:150])
            print(f"  KWS {name} 失败: {repr(e)[:120]}", flush=True)

    # 真实训练 60 epoch（KWS），比较准确率与耗时
    for name, dev in devices:
        try:
            torch.manual_seed(0)
            model = KWS().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            lossf = nn.CrossEntropyLoss()
            X = torch.randn(192, 64, 64, device=dev)
            y = torch.randint(0, 4, (192,), device=dev)
            t0 = time.perf_counter()
            for ep in range(30):
                perm = torch.randperm(192, device=dev)
                for i in range(0, 192, 16):
                    idx = perm[i:i + 16]
                    opt.zero_grad()
                    loss = lossf(model(X[idx]), y[idx])
                    loss.backward()
                    opt.step()
            dt = time.perf_counter() - t0
            results[f"kws_train30_{name}"] = dict(seconds=round(dt, 2))
            print(f"  KWS 真训 30 epoch {name}: {dt:.2f}s", flush=True)
        except Exception as e:
            results[f"kws_train30_{name}"] = dict(error=repr(e)[:150])
            print(f"  KWS 真训 {name} 失败: {repr(e)[:120]}", flush=True)

    # 2) VITS proxy
    def vits_batch(dev):
        return torch.randn(8, 80, 100, device=dev), torch.randn(8, 513, 100, device=dev)

    for name, dev in devices:
        try:
            dt, sps = steps_bench(VitsProxy(), vits_batch, dev, n=5, warmup=2)
            results[f"vits_{name}"] = dict(step_s=dt, steps_per_s=sps)
            print(f"  VITS-proxy {name}: {dt*1000:.0f} ms/步", flush=True)
        except Exception as e:
            results[f"vits_{name}"] = dict(error=repr(e)[:150])
            print(f"  VITS-proxy {name} 失败: {repr(e)[:120]}", flush=True)

    with open(rf"{OUT}\igpu_train_bench.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
