# -*- coding: utf-8 -*-
"""audio_tts_bench.py — TTS 侧的可用测量：
  1) Windows SAPI（本机现成、非神经网络）合成速度；
  2) VITS 量级模型（≈19M 参数）的前向耗时 → 估算神经网络 TTS 的 RTF 下限。
（Kokoro/Piper 的 pip 安装在慢网下没跑完，本轮不给质量结论，只给算力口径。）
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import psutil
import soundfile as sf
import torch
import torch.nn as nn

OUT = r"D:\work\research\bench_results\audio_lab"
torch.set_num_threads(6)


def rss():
    return psutil.Process().memory_info().rss / 1e9


def sapi_bench(text, reps=3):
    """用 PowerShell 调 SAPI 合成，计时。"""
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$out = Join-Path $env:TEMP 'sapi_bench.wav';"
        "$s.SetOutputToWaveFile($out);"
        "$s.Speak($env:SAPI_TEXT);"
        "$s.SetOutputToNull(); $s.Dispose()"
    )
    env = dict(os.environ, SAPI_TEXT=text)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       env=env, capture_output=True, timeout=120)
        times.append(time.perf_counter() - t0)
    tmp = os.path.join(os.environ.get("TEMP", "."), "sapi_bench.wav")
    d, sr = sf.read(tmp)
    dur = len(d) / sr
    best = min(times)
    return dict(audio_s=round(dur, 2), wall_s=round(best, 2), rtf=round(best / dur, 3),
                speed_x=round(dur / best, 2), reps=reps)


class VitsLike(nn.Module):
    """VITS 量级（约 19M）的前向代理：文本嵌入 → conv 堆栈 → mel 上采样 → 波形。"""

    def __init__(self, ch=512, blocks=6, n_mels=80):
        super().__init__()
        self.emb = nn.Embedding(256, ch)
        self.pre = nn.Conv1d(ch, ch, 5, padding=2)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(),
                          nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(),
                          nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU())
            for _ in range(blocks)])
        self.up = nn.ModuleList([nn.ConvTranspose1d(ch, ch, 4, 2, 1) for _ in range(10)])  # 2^10=1024，30 音素≈1.9s
        self.hifi = nn.ModuleList([nn.Conv1d(ch, ch, 5, padding=2) for _ in range(6)])
        self.to_mel = nn.Conv1d(ch, n_mels, 1)
        self.to_wav = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, tok):
        h = self.emb(tok).transpose(1, 2)
        h = self.pre(h)
        for b in self.blocks:
            h = h + b(h)
        for u in self.up:
            h = nn.functional.leaky_relu(u(h), 0.1)
        for c in self.hifi:
            h = nn.functional.leaky_relu(c(h), 0.1)
        return self.to_mel(h), self.to_wav(h)


def vits_inference_bench(seconds=2.0):
    m = VitsLike().eval()
    n = sum(p.numel() for p in m.parameters())
    # 音素率约 15/s，上采样 2^4=16 后 ≈ 240 帧/s（简化）
    n_tok = int(15 * seconds)
    tok = torch.randint(1, 200, (1, n_tok))
    with torch.no_grad():
        m(tok)                      # warmup
        t0 = time.perf_counter()
        mel, wav = m(tok)
        dt = time.perf_counter() - t0
    audio_len = wav.shape[-1] / 16000
    return dict(params=n, tokens=n_tok, forward_s=round(dt, 3),
                est_audio_s=round(float(audio_len), 2),
                rtf=round(dt / max(audio_len, 1e-6), 3), rss_gb=round(rss(), 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    text = ("Artificial intelligence models are getting smaller and faster every year. "
            "On a modern laptop without a graphics card, some of these models can still run in real time.")
    sapi = sapi_bench(text)
    print(f"  SAPI: 音频 {sapi['audio_s']}s | 合成 {sapi['wall_s']}s | RTF {sapi['rtf']}"
          f"（{sapi['speed_x']}× 实时）", flush=True)

    vits = vits_inference_bench()
    print(f"  VITS-like(19M): {vits['forward_s']}s 前向 → 估算音频 {vits['est_audio_s']}s | "
          f"RTF≈{vits['rtf']}", flush=True)

    rows = dict(sapi=sapi, vits_like=vits)
    with open(os.path.join(OUT, "tts_report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
