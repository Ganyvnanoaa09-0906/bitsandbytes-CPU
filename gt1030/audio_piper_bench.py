# -*- coding: utf-8 -*-
"""audio_piper_bench.py — Piper（ONNX，CPU TTS）实测：加载/合成速度、音质（用 ASR 回环算 WER）。"""
from __future__ import annotations

import json
import os
import time
import wave

import numpy as np
import psutil

OUT = r"D:\work\research\bench_results\audio_lab"
VOICES = {
    "en_lessac": r"D:\work\textmodel\piper-voices\en_US-lessac-medium.onnx",
    "zh_huayan": r"D:\work\textmodel\piper-voices\zh_CN-huayan-medium.onnx",
}

TEXTS = {
    "en_lessac": [
        ("short", "Hello there. This is a test of the Piper text to speech system."),
        ("long", "Artificial intelligence models are getting smaller and faster every year. "
                 "On a modern laptop without a graphics card, some of these models can still run in real time. "
                 "In this test we measure how much audio a small speech synthesis model can generate in one second "
                 "of wall clock time. The result is usually expressed as a real time factor."),
    ],
    "zh_huayan": [
        ("short", "你好，这是一段语音合成的测试。"),
        ("long", "人工智能模型每年都在变得更小更快。在没有独立显卡的普通笔记本上，一些模型依然可以实时运行。"
                 "在这个测试里，我们测量一个小的语音合成模型在一秒钟内能生成多少音频。结果通常用实时率来表示。"),
    ],
}


def rss():
    return psutil.Process().memory_info().rss / 1e9


def synth(voice, text, path):
    t0 = time.perf_counter()
    with wave.open(path, "wb") as w:
        voice.synthesize_wav(text, w)      # 1.8 的 API：会自己写好 WAV 头
    dt = time.perf_counter() - t0
    with wave.open(path, "rb") as r:
        dur = r.getnframes() / float(r.getframerate())
        sr = r.getframerate()
    return dt, dur, sr


def main():
    from piper import PiperVoice

    rows = []
    for tag, path in VOICES.items():
        if not os.path.exists(path):
            print(f"  [skip] {tag} 缺模型"); continue
        t0 = time.perf_counter()
        voice = PiperVoice.load(path)
        load_s = time.perf_counter() - t0
        print(f"  {tag}: 加载 {load_s:.1f}s | RSS {rss():.2f} GB", flush=True)
        for label, text in TEXTS[tag]:
            out = os.path.join(OUT, f"piper_{tag}_{label}.wav")
            dt, dur, sr = synth(voice, text, out)
            rows.append(dict(voice=tag, kind=label, chars=len(text), audio_s=round(dur, 2),
                             wall_s=round(dt, 2), rtf=round(dt / max(dur, 1e-6), 3),
                             speed_x=round(dur / max(dt, 1e-6), 2), sr=sr, path=out,
                             load_s=round(load_s, 2)))
            print(f"    {label}: {len(text)} 字 → 音频 {dur:.2f}s | 合成 {dt:.2f}s | "
                  f"RTF {dt/max(dur,1e-6):.3f}（{dur/max(dt,1e-6):.1f}× 实时）| {sr} Hz", flush=True)

    # ASR 回环：用 faster-whisper 检查英文可懂度
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(r"D:\work\textmodel\faster-whisper-small", device="cpu",
                         compute_type="int8", cpu_threads=6)
        for r in rows:
            if r["voice"] != "en_lessac":
                continue
            segs, _ = m.transcribe(r["path"], language="en", beam_size=1, temperature=0.0)
            text = " ".join(s.text.strip() for s in segs)
            ref = dict(TEXTS["en_lessac"])[r["kind"]]
            import re
            a = re.findall(r"[a-z']+", ref.lower())
            b = re.findall(r"[a-z']+", text.lower())
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
            r["asr_text"] = text.strip()[:200]
            r["wer"] = round(wer, 4)
            print(f"    {r['kind']} ASR 回环 WER {wer:.3f} | {text.strip()[:100]}", flush=True)
    except Exception as e:
        print("  ASR 回环失败:", repr(e)[:100])

    with open(os.path.join(OUT, "piper_report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
