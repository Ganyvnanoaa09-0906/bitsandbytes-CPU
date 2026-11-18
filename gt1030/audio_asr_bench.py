# -*- coding: utf-8 -*-
"""audio_asr_bench.py — faster-whisper 在 CPU 上的推理实测（RTF / 内存 / 与原文对比）。"""
from __future__ import annotations

import json
import os
import re
import time

import psutil
import soundfile as sf
from faster_whisper import WhisperModel

MODEL = r"D:\work\textmodel\faster-whisper-small"
AUDIO = r"D:\work\research\bench_results\audio_lab"
REF = {
    "sapi_10s": "The quick brown fox jumps over the lazy dog. This sentence is used to test speech "
                "recognition accuracy on a small laptop.",
    "sapi_30s": "Artificial intelligence models are getting smaller and faster every year. On a modern "
                "laptop without a graphics card, some of these models can still run in real time. In this "
                "test we measure how much audio a small speech recognizer can transcribe in one second of "
                "wall clock time. The result is usually expressed as a real time factor, where one means the "
                "model processes audio as fast as it plays.",
}


def wer(ref, hyp):
    r = re.findall(r"[a-z']+", ref.lower())
    h = re.findall(r"[a-z']+", hyp.lower())
    # 简单编辑距离
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


def rss():
    return psutil.Process().memory_info().rss / 1e9


def main():
    t0 = time.time()
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=6)
    load_s = time.time() - t0
    print(f"  模型加载 {load_s:.1f}s | RSS {rss():.2f} GB", flush=True)

    rows = []
    for name in ("sapi_10s", "sapi_30s"):
        p = os.path.join(AUDIO, name + ".wav")
        if not os.path.exists(p):
            continue
        d, sr = sf.read(p)
        dur = len(d) / sr
        t1 = time.time()
        segments, info = model.transcribe(p, language="en", beam_size=1, temperature=0.0)
        text = " ".join(s.text.strip() for s in segments)
        dt = time.time() - t1
        w = wer(REF[name], text) if name in REF else None
        rows.append(dict(name=name, audio_s=round(dur, 2), wall_s=round(dt, 2),
                         rtf=round(dt / dur, 3), speed_x=round(dur / dt, 2),
                         wer=round(w, 4) if w is not None else None,
                         text=text.strip()[:160]))
        print(f"  {name}: 音频 {dur:.1f}s | 转写 {dt:.1f}s | RTF {dt/dur:.3f}（{dur/dt:.2f}× 实时）"
              f" | WER {w:.3f}", flush=True)
        print(f"    {text.strip()[:150]}", flush=True)

    # 长音频：把 30s 拼成 5 分钟，测吞吐
    d, sr = sf.read(os.path.join(AUDIO, "sapi_30s.wav"))
    long = os.path.join(AUDIO, "long_5min.wav")
    if not os.path.exists(long):
        import numpy as np
        reps = int(300 / (len(d) / sr)) + 1
        sf.write(long, np.tile(d, reps), sr)
    d2, sr2 = sf.read(long)
    dur2 = len(d2) / sr2
    t2 = time.time()
    segments, _ = model.transcribe(long, language="en", beam_size=1, temperature=0.0)
    n_seg = sum(1 for _ in segments)
    dt2 = time.time() - t2
    rows.append(dict(name="long_5min", audio_s=round(dur2, 1), wall_s=round(dt2, 1),
                     rtf=round(dt2 / dur2, 3), speed_x=round(dur2 / dt2, 2), segs=n_seg))
    print(f"  long_5min: 音频 {dur2:.0f}s | 转写 {dt2:.1f}s | RTF {dt2/dur2:.3f}"
          f"（{dur2/dt2:.2f}×）| {n_seg} 段 | RSS {rss():.2f} GB", flush=True)

    out = os.path.join(AUDIO, "asr_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(model="faster-whisper-small int8", cpu_threads=6,
                       load_s=round(load_s, 1), rows=rows), f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
