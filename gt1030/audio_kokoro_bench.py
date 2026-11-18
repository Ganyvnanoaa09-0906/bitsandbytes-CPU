# -*- coding: utf-8 -*-
"""audio_kokoro_bench.py — Kokoro-82M（pytorch 权重，CPU）TTS 实测 + ASR 回环。

Kokoro 82M：小、快、自然度在同等规模里很强；这里用本地权重（ModelScope 下载）离线跑。
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import psutil
import soundfile as sf


OUT = r"D:\work\research\bench_results\audio_lab"
KOKORO = r"D:\work\textmodel\kokoro-82m"
VOICES = {
    "af_heart": os.path.join(KOKORO, "voices", "af_heart.pt"),
    "am_michael": os.path.join(KOKORO, "voices", "am_michael.pt"),
}
TEXTS = [
    ("short", "Hello there. This is a test of the Kokoro text to speech system."),
    ("long", "Artificial intelligence models are getting smaller and faster every year. "
             "On a modern laptop without a graphics card, some of these models can still run in real time. "
             "In this test we measure how much audio a small speech synthesis model can generate in one second "
             "of wall clock time. The result is usually expressed as a real time factor."),
]


def rss():
    return psutil.Process().memory_info().rss / 1e9


def main():
    import torch
    from kokoro import KModel, KPipeline

    torch.set_num_threads(6)
    t0 = time.perf_counter()
    # 用本地权重（避开联网）
    model = KModel(repo_id=None, model=os.path.join(KOKORO, "kokoro-v1_0.pth"),
                   config=os.path.join(KOKORO, "config.json")).eval()
    load_s = time.perf_counter() - t0
    print(f"  KModel 加载 {load_s:.1f}s | RSS {rss():.2f} GB", flush=True)

    pipe = KPipeline(lang_code="a", repo_id=None, model=model)
    rows = []
    for vtag, vpath in VOICES.items():
        if not os.path.exists(vpath):
            continue
        rep = torch.load(vpath, map_location="cpu", weights_only=True)
        for label, text in TEXTS:
            t1 = time.perf_counter()
            chunks = []
            for _, _, audio in pipe(text, voice=rep, speed=1.0):
                chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
            dt = time.perf_counter() - t1
            audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
            sr = 24000
            dur = len(audio) / sr
            out = os.path.join(OUT, f"kokoro_{vtag}_{label}.wav")
            sf.write(out, audio, sr)
            rows.append(dict(voice=vtag, kind=label, audio_s=round(dur, 2), wall_s=round(dt, 2),
                             rtf=round(dt / max(dur, 1e-6), 3), speed_x=round(dur / max(dt, 1e-6), 2),
                             sr=sr, path=out, load_s=round(load_s, 1)))
            print(f"  {vtag}/{label}: 音频 {dur:.2f}s | 合成 {dt:.2f}s | RTF {dt/max(dur,1e-6):.3f}"
                  f"（{dur/max(dt,1e-6):.1f}× 实时）", flush=True)

    # ASR 回环
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(r"D:\work\textmodel\faster-whisper-small", device="cpu",
                         compute_type="int8", cpu_threads=6)
        for r in rows:
            segs, _ = m.transcribe(r["path"], language="en", beam_size=1, temperature=0.0)
            got = " ".join(s.text.strip() for s in segs)
            ref = dict(TEXTS)[r["kind"]]
            a = re.findall(r"[a-z']+", ref.lower())
            b = re.findall(r"[a-z']+", got.lower())
            d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
            for i in range(len(a) + 1):
                d[i][0] = i
            for j in range(len(b) + 1):
                d[0][j] = j
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                                  d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            wer = d[len(a)][len(b)] / max(1, len(a))
            r["asr_text"] = got.strip()[:200]
            r["wer"] = round(wer, 4)
            print(f"    {r['voice']}/{r['kind']} ASR WER {wer:.3f} | {got.strip()[:100]}", flush=True)
    except Exception as e:
        print("  ASR 回环失败:", repr(e)[:120])

    with open(os.path.join(OUT, "kokoro_report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
