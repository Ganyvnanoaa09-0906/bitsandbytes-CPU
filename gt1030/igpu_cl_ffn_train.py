# -*- coding: utf-8 -*-
"""igpu_cl_ffn_train.py — iGPU(OpenCL/CLBlast) 跑 Transformer FFN 训练步，对比 CPU。

为什么测这个：卷积类音频模型（KWS/VITS）走 im2col+GEMM 反而更慢（见 igpu_cl_audio_conv.py），
但 transformer 类的音频模型（AST / Whisper 编码器 / Audio-MAE）计算主要是**大 GEMM**，
这正是 CLBlast 的强项（2048³ 时核显 500 GFLOPS vs CPU 226 GFLOPS）。

这里手工实现一个 2 层 FFN（768→3072→768）的 fwd+bwd，GEMM 全部走核显，逐元素在 CPU，
和纯 CPU torch 版本比单步耗时；形状取音频 transformer 的典型值。
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
OUT = r"D:\work\research\bench_results\audio_lab"
torch.set_num_threads(6)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    c = np.sqrt(2 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * c * (1 + 3 * 0.044715 * x ** 2)


def step_igpu(x, W1, b1, W2, b2, tgt, G):
    # forward（GEMM 走核显）
    h1 = G.sgemm(x, W1) + b1
    a = gelu(h1)
    h2 = G.sgemm(a, W2) + b2
    diff = h2 - tgt
    loss = float((diff ** 2).mean())
    # backward
    dh2 = 2.0 * diff / diff.size
    dW2 = G.sgemm(G.transpose_f32(a), dh2)    # a^T @ dh2
    db2 = dh2.sum(0)
    da = G.sgemm_nt(dh2, W2)          # dh2 @ W2^T
    dh1 = da * gelu_grad(h1)
    dW1 = G.sgemm(G.transpose_f32(x), dh1)    # x^T @ dh1
    db1 = dh1.sum(0)
    # SGD 更新（CPU）
    lr = 1e-3
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return loss


def step_cpu(x, W1, b1, W2, b2, tgt):
    xt, W1t, W2t = torch.from_numpy(x), W1, W2
    h1 = xt @ W1t + b1
    a = torch.nn.functional.gelu(h1)
    h2 = a @ W2t + b2
    diff = h2 - tgt
    loss = float((diff ** 2).mean())
    loss_t = (diff ** 2).mean()
    grads = torch.autograd.grad(loss_t, (W1t, b1, W2t, b2))
    lr = 1e-3
    with torch.no_grad():
        W1t -= lr * grads[0]
        b1 -= lr * grads[1]
        W2t -= lr * grads[2]
        b2 -= lr * grads[3]
    return loss


def main():
    import gt1030.hybrid_gpu as G
    G.init()
    res = {}
    for dim, ff, bs in ((768, 3072, 32), (1024, 4096, 32), (768, 3072, 128)):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((bs, dim)).astype(np.float32) * 0.1
        W1 = (rng.standard_normal((dim, ff)) * 0.05).astype(np.float32)
        b1 = np.zeros(ff, np.float32)
        W2 = (rng.standard_normal((ff, dim)) * 0.05).astype(np.float32)
        b2 = np.zeros(dim, np.float32)
        tgt = rng.standard_normal((bs, dim)).astype(np.float32)

        # CPU
        W1c = torch.from_numpy(W1.copy()).requires_grad_(True)
        b1c = torch.from_numpy(b1.copy()).requires_grad_(True)
        W2c = torch.from_numpy(W2.copy()).requires_grad_(True)
        b2c = torch.from_numpy(b2.copy()).requires_grad_(True)
        step_cpu(x, W1c, b1c, W2c, b2c, torch.from_numpy(tgt))
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            step_cpu(x, W1c, b1c, W2c, b2c, torch.from_numpy(tgt))
            ts.append(time.perf_counter() - t)
        cpu = min(ts) * 1e3

        # iGPU
        W1g, b1g, W2g, b2g = W1.copy(), b1.copy(), W2.copy(), b2.copy()
        step_igpu(x, W1g, b1g, W2g, b2g, tgt, G)
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            step_igpu(x, W1g, b1g, W2g, b2g, tgt, G)
            ts.append(time.perf_counter() - t)
        gpu = min(ts) * 1e3
        key = f"ffn_{dim}x{ff}_bs{bs}"
        res[key] = dict(cpu_ms=round(cpu, 2), igpu_ms=round(gpu, 2), speedup=round(cpu / gpu, 2))
        print(f"  {key:<22} CPU {cpu:7.2f} ms | iGPU {gpu:7.2f} ms | {cpu/gpu:5.2f}×", flush=True)

    with open(rf"{OUT}\cl_ffn_train.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
