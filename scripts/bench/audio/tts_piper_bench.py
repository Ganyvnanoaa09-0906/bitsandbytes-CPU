# -*- coding: utf-8 -*-
"""tts_piper_bench.py — Piper（ONNX，CPU TTS）实测：加载/合成速度与实时率。

依赖（不入库）：
    pip install piper-tts soundfile psutil
权重（不入库）：
    --voice 指向 *.onnx 及其同名 *.onnx.json（Piper 语音包）

用法：
    python scripts/bench/audio/tts_piper_bench.py --voice voices/en_US-lessac-medium.onnx \\
        --out out/piper.json
    python ... --smoke          # 只合成一句短文本

输出 JSON（schema v1）：加载耗时、每句合成耗时、音频时长、RTF、字符/秒。
RTF = 音频时长 / 墙钟耗时（>1 表示快于实时）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

SCHEMA = 1

SHORT = "Hello there. This is a test of the Piper text to speech system."
LONG = ("Artificial intelligence models are getting smaller and faster every year. "
        "On a modern laptop without a graphics card, some of these models can still run "
        "in real time. In this test we measure how much audio a small speech synthesis "
        "model can generate in one second of wall clock time.")


def env_fingerprint(threads: int) -> dict:
    fp = {"python": sys.version.split()[0], "platform": platform.platform(),
          "machine": platform.machine(), "threads": threads}
    try:
        import psutil
        fp["cpu_count"] = psutil.cpu_count(logical=True)
        fp["rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 3)
    except Exception:
        pass
    try:
        import piper
        fp["piper"] = getattr(piper, "__version__", "?")
    except Exception:
        pass
    return fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="Piper *.onnx 语音包路径")
    ap.add_argument("--out", default="", help="JSON 输出路径")
    ap.add_argument("--wav-dir", default="", help="可选：把合成音频写到该目录")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="只合成一句短文本")
    a = ap.parse_args()

    try:
        from piper import PiperVoice
    except Exception as e:
        print(f"[skip] 缺少依赖 piper-tts: {e}", file=sys.stderr)
        return 2
    if not os.path.exists(a.voice):
        print(f"[skip] 语音包不存在: {a.voice}", file=sys.stderr)
        return 2

    texts = [("short", SHORT)] if a.smoke else [("short", SHORT), ("long", LONG)]
    t0 = time.time()
    voice = PiperVoice.load(a.voice)
    load_s = time.time() - t0

    if a.wav_dir:
        os.makedirs(a.wav_dir, exist_ok=True)
    rows = []
    for label, text in texts:
        wav = os.path.join(a.wav_dir, f"piper_{label}.wav") if a.wav_dir else None
        t1 = time.time()
        # piper-tts >= 1.2：synthesize_wav(text, wav_file) / 旧版 synthesize(text)
        if wav and hasattr(voice, "synthesize_wav"):
            with open(wav, "wb") as wf:
                voice.synthesize_wav(text, wf)
            synth_s = time.time() - t1
            import wave
            with wave.open(wav, "rb") as wf:
                audio_s = wf.getnframes() / float(wf.getframerate())
        else:
            chunks = list(voice.synthesize(text))
            synth_s = time.time() - t1
            audio_s = sum(len(c.audio_int16_bytes) for c in chunks) / 2.0 / float(
                getattr(chunks[0], "sample_rate", 22050)) if chunks else 0.0
        row = {"label": label, "chars": len(text), "audio_s": round(audio_s, 3),
               "wall_s": round(synth_s, 3),
               "rtf": round(audio_s / synth_s, 2) if synth_s > 0 else None,
               "chars_per_s": round(len(text) / synth_s, 1) if synth_s > 0 else None,
               "wav": wav}
        rows.append(row)
        print(f"  {label}: {len(text)} 字符 -> {audio_s:.2f}s 音频 / {synth_s:.3f}s 墙钟 "
              f"RTF={row['rtf']}", flush=True)

    rtfs = [r["rtf"] for r in rows if r["rtf"]]
    rep = {"schema_version": SCHEMA, "kind": "tts_piper_bench",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "env": env_fingerprint(a.threads),
           "args": {"voice": os.path.basename(a.voice), "smoke": a.smoke},
           "load_s": round(load_s, 3), "results": rows,
           "summary": {"mean_rtf": round(sum(rtfs) / len(rtfs), 2) if rtfs else None}}
    print("  " + json.dumps(rep["summary"], ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
