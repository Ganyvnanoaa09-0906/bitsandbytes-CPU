# -*- coding: utf-8 -*-
"""igpu_audio_train_bench3.py — 真训练对比：同一个 KWS 数据集，CPU vs iGPU(bs=64) vs iGPU(fp16)。

用合成 log-mel（可复现），训练 60 epoch，比较：耗时、最终 loss、吞吐。
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


def make_data(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    # 合成 4 类的 log-mel：每类有不同的频带模式，保证可学
    X, y = [], []
    for c in range(4):
        for _ in range(n // 4):
            base = torch.zeros(64, 64)
            lo = 8 + c * 12
            base[lo:lo + 10, :] = 1.0
            base += 0.35 * torch.randn(64, 64, generator=g)
            X.append(base)
            y.append(c)
    return torch.stack(X), torch.tensor(y)


def train_run(dev, dtype, bs, epochs=60, seed=1):
    torch.manual_seed(seed)
    X, y = make_data()
    X, y = X.to(dev, dtype=dtype), y.to(dev)
    m = KWS().to(dev, dtype=dtype)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    last = None
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        tot, corr = 0, 0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = m(X[idx].to(dtype))
            loss = lf(out.float(), y[idx])
            loss.backward()
            opt.step()
            last = float(loss)
            corr += int((out.argmax(1) == y[idx]).sum())
            tot += len(idx)
    dt = time.perf_counter() - t0
    return dict(seconds=round(dt, 2), final_loss=round(last, 4) if last else None,
                train_acc=round(corr / tot, 3), epochs=epochs, bs=bs, dtype=str(dtype))


def main():
    res = {}
    jobs = [("cpu_fp32_bs64", torch.device("cpu"), torch.float32, 64)]
    if HAS_DML:
        dev = dml.device(0)
        jobs += [("igpu_fp32_bs64", dev, torch.float32, 64),
                 ("igpu_fp32_bs192", dev, torch.float32, 192),
                 ("igpu_fp16_bs64", dev, torch.float16, 64)]
    for name, dev, dtype, bs in jobs:
        try:
            r = train_run(dev, dtype, bs)
            res[name] = r
            print(f"  {name:<16} {r['seconds']:6.2f}s | loss {r['final_loss']} | "
                  f"train_acc {r['train_acc']}", flush=True)
        except Exception as e:
            res[name] = dict(error=repr(e)[:150])
            print(f"  {name} 失败: {repr(e)[:120]}", flush=True)
    with open(rf"{OUT}\igpu_train_bench3.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
