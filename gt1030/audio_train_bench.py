# -*- coding: utf-8 -*-
"""audio_train_bench.py — CPU 上音频模型"训练"的可行性实测。

三档：
  1) 微型 CNN（唤醒词级别，~0.1M 参数）：真正跑 fwd+bwd，测 steps/s；
  2) VITS（TTS 声学模型，torchaudio 自带结构，量级 ≈ 20–30M）：测一步耗时，外推微调成本；
  3) Whisper-small 量级（transformers 按 config 建，≈244M）：测一步，外推 ASR 微调成本。
输出每档的单步耗时、显存/RSS、以及"N 步需要多少小时"。
"""
from __future__ import annotations

import json
import time

import numpy as np
import psutil
import torch
import torch.nn as nn

torch.set_num_threads(6)


def rss():
    return psutil.Process().memory_info().rss / 1e9


class TinyWakeWord(nn.Module):
    """1 秒 16 kHz → 64 mel → 2 层 CNN → 2 分类（唤醒/非唤醒）。"""

    def __init__(self, n_mels=64, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def bench_steps(make_batch, model, n_steps=30, warmup=5, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters())
    # 预热
    for _ in range(warmup):
        x, y = make_batch()
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x, y = make_batch()
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
    dt = (time.perf_counter() - t0) / n_steps
    return dict(params=params, step_s=round(dt, 3), steps_per_s=round(1 / dt, 3), rss_gb=round(rss(), 2))


def main():
    rows = {}
    # 1) 微型 CNN
    m = TinyWakeWord()
    rows["tiny_cnn"] = bench_steps(lambda: (torch.randn(16, 64, 98), torch.randint(0, 2, (16,))), m)
    print("  tiny_cnn:", rows["tiny_cnn"], flush=True)

    # 2) VITS 代理（torchaudio 2.11 已移除 VITS；用同量级 conv1d 堆栈量吞吐）
    try:
        class VitsProxy(nn.Module):
            def __init__(self, ch=512, layers=8):
                super().__init__()
                self.inp = nn.Conv1d(80, ch, 3, padding=1)
                self.blocks = nn.ModuleList([
                    nn.Sequential(nn.Conv1d(ch, ch * 2, 3, padding=1), nn.GLU(1),
                                  nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())
                    for _ in range(layers)])
                self.out = nn.Conv1d(ch, 513, 1)

            def forward(self, x):
                h = self.inp(x)
                for b in self.blocks:
                    h = h + b(h)
                return self.out(h)

        v = VitsProxy()
        n = sum(p.numel() for p in v.parameters())
        opt = torch.optim.AdamW(v.parameters(), lr=1e-4)
        mel_in = torch.randn(8, 80, 100)     # batch 8 × 1 秒
        target = torch.randn(8, 513, 100)
        for _ in range(2):
            opt.zero_grad(); loss = (v(mel_in) - target).pow(2).mean(); loss.backward(); opt.step()
        t0 = time.perf_counter()
        for _ in range(3):
            opt.zero_grad(); loss = (v(mel_in) - target).pow(2).mean(); loss.backward(); opt.step()
        step = (time.perf_counter() - t0) / 3
        rows['vits_proxy'] = dict(params=n, step_s=round(step, 2), batch=8, rss_gb=round(rss(), 2))
        print('  vits_proxy:', rows['vits_proxy'], flush=True)
    except Exception as e:
        rows['vits_proxy'] = dict(error=repr(e)[:120])
        print('  vits_proxy 失败:', repr(e)[:120], flush=True)

    # 3) Whisper-small 量级（按 config 建，随机权重，只量训练吞吐）
    try:
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        cfg = WhisperConfig(vocab_size=51865, d_model=768, encoder_layers=12,
                            encoder_attention_heads=12, decoder_layers=12,
                            decoder_attention_heads=12)
        w = WhisperForConditionalGeneration(cfg)
        n = sum(p.numel() for p in w.parameters())
        opt = torch.optim.AdamW(w.parameters(), lr=1e-5)
        mel = torch.randn(2, 80, 3000)                 # 2 条 30 秒 mel
        labels = torch.randint(0, 100, (2, 20))
        t0 = time.perf_counter()
        opt.zero_grad()
        out = w(input_features=mel, labels=labels)
        out.loss.backward()
        opt.step()
        step = time.perf_counter() - t0
        rows["whisper_small"] = dict(params=n, step_s=round(step, 2), batch=2, rss_gb=round(rss(), 2))
        print("  whisper_small:", rows["whisper_small"], flush=True)
    except Exception as e:
        rows["whisper_small"] = dict(error=repr(e)[:120])
        print("  whisper_small 失败:", repr(e)[:120], flush=True)

    # 外推
    proj = {}
    for k, v in rows.items():
        if "step_s" in v:
            proj[k] = dict(k10_steps_min=round(v["step_s"] * 10000 / 60, 1),
                           k100_steps_h=round(v["step_s"] * 100000 / 3600, 1))
    print("  外推（1e4 / 1e5 步）:", proj, flush=True)
    out = r"D:\work\research\bench_results\audio_lab\train_bench.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, projection=proj), f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
