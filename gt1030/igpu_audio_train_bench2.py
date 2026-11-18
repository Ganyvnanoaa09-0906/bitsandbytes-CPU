# -*- coding: utf-8 -*-
"""igpu_audio_train_bench2.py — 核显训练第二轮：加大 batch（摊薄开销）、Whisper-small 量级、fp16。

假设：核显在 batch 很小时输在 kernel 启动开销；Vega 8 只有 6 CU 且与 CPU 共享 DDR4 带宽，
所以只有「计算密度高」的层（大 GEMM）才可能赢。这里逐条验证。
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


def bench(fn, n=8, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    devs = [("cpu", torch.device("cpu"))]
    if HAS_DML:
        devs.append(("igpu", dml.device(0)))
    res = {}

    # 1) 大 batch 的 KWS
    print("== KWS（不同 batch）", flush=True)
    for bs in (16, 64, 256):
        for name, dev in devs:
            m = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 4)).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            x = torch.randn(bs, 1, 64, 64, device=dev)
            y = torch.randint(0, 4, (bs,), device=dev)
            lf = nn.CrossEntropyLoss()

            def step():
                opt.zero_grad()
                lf(m(x), y).backward()
                opt.step()

            dt = bench(step)
            res[f"kws_bs{bs}_{name}"] = round(dt, 4)
            print(f"  bs={bs:<4} {name:<4} {dt*1000:7.1f} ms/步  ({bs/dt:7.0f} 样本/s)", flush=True)

    # 2) 大 GEMM 的训练步（模拟 transformer FFN）
    print("== 大 Linear（FFN 规模）", flush=True)
    for dim, bs in ((2048, 8), (4096, 8), (4096, 32)):
        for name, dev in devs:
            m = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            x = torch.randn(bs, dim, device=dev)
            tgt = torch.randn(bs, dim, device=dev)

            def step():
                opt.zero_grad()
                ((m(x) - tgt) ** 2).mean().backward()
                opt.step()

            dt = bench(step, n=5)
            res[f"ffn_{dim}x{bs}_{name}"] = round(dt, 4)
            print(f"  dim={dim:<5} bs={bs:<3} {name:<4} {dt*1000:7.1f} ms/步", flush=True)

    # 3) fp16 对比（DirectML 支持 fp16）
    print("== fp16（大 Linear）", flush=True)
    if HAS_DML:
        for name, dev in devs:
            try:
                m = nn.Sequential(nn.Linear(4096, 16384), nn.GELU(), nn.Linear(16384, 4096)).to(dev).half()
                opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
                x = torch.randn(8, 4096, device=dev, dtype=torch.float16)
                tgt = torch.randn(8, 4096, device=dev, dtype=torch.float16)

                def step():
                    opt.zero_grad()
                    ((m(x) - tgt) ** 2).mean().backward()
                    opt.step()

                dt = bench(step, n=5)
                res[f"ffn_fp16_{name}"] = round(dt, 4)
                print(f"  fp16 {name:<4} {dt*1000:7.1f} ms/步", flush=True)
            except Exception as e:
                res[f"ffn_fp16_{name}"] = dict(error=repr(e)[:120])
                print(f"  fp16 {name} 失败: {repr(e)[:100]}", flush=True)

    with open(rf"{OUT}\igpu_train_bench2.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
