# -*- coding: utf-8 -*-
"""tts_kokoro_bench.py — Kokoro-82M（PyTorch 权重，CPU TTS）实测：加载/合成速度与实时率。

依赖（不入库）：
    pip install kokoro soundfile torch psutil
权重（不入库）：
    --kokoro-dir 指向 Kokoro-82M 目录（含 config.json、*.pth、voices/*.pt）
    --voice 指向 voices/ 下的 .pt（如 af_heart.pt）

用法：
    python scripts/bench/audio/tts_kokoro_bench.py --kokoro-dir models/kokoro-82m \\
        --voice af_heart.pt --out out/kokoro.json
    python ... --smoke          # 只合成一句短文本

输出 JSON（schema v1）：加载耗时、每句合成耗时、音频时长、RTF。
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

SHORT = "Hello there. This is a test of the Kokoro text to speech system."
LONG = ("Artificial intelligence models are getting smaller and faster every year. "
        "On a modern laptop without a graphics card, some of these models can still run "
        "in real time. In this test we measure how much audio a small synthesis model can "
        "produce per second of wall clock time.")


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
        import torch
        fp["torch"] = torch.__version__
    except Exception:
        pass
    return fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kokoro-dir", required=True, help="Kokoro-82M 权重目录")
    ap.add_argument("--voice", default="af_heart.pt", help="voices/ 下的音色文件名")
    ap.add_argument("--out", default="", help="JSON 输出路径")
    ap.add_argument("--wav-dir", default="", help="可选：写出合成音频")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="只合成一句短文本")
    a = ap.parse_args()

    try:
        import numpy as np
        import torch
        from kokoro import KModel, KPipeline
    except Exception as e:
        print(f"[skip] 缺少依赖（kokoro/torch/numpy）: {e}", file=sys.stderr)
        return 2
    if not os.path.isdir(a.kokoro_dir):
        print(f"[skip] 权重目录不存在: {a.kokoro_dir}", file=sys.stderr)
        return 2
    torch.set_num_threads(a.threads)

    texts = [("short", SHORT)] if a.smoke else [("short", SHORT), ("long", LONG)]
    # 完全离线加载：KModel 直接吃本地 .pth + config.json，避免 HF Hub 访问
    import glob
    pth = os.path.join(a.kokoro_dir, "kokoro-v1_0.pth")
    if not os.path.exists(pth):
        cands = glob.glob(os.path.join(a.kokoro_dir, "*.pth"))
        if not cands:
            print(f"[skip] {a.kokoro_dir} 下找不到 *.pth", file=sys.stderr)
            return 2
        pth = cands[0]
    cfg = os.path.join(a.kokoro_dir, "config.json")
    t0 = time.time()
    kmodel = KModel(repo_id=None, model=pth, config=cfg if os.path.exists(cfg) else None).eval()
    pipe = KPipeline(lang_code="a", repo_id=None, model=kmodel)
    load_s = time.time() - t0

    if a.wav_dir:
        os.makedirs(a.wav_dir, exist_ok=True)
    rows = []
    for label, text in texts:
        t1 = time.time()
        vp = a.voice if os.path.exists(a.voice) else os.path.join(a.kokoro_dir, "voices", a.voice)
        chunks = list(pipe(text, voice=vp))
        synth_s = time.time() - t1
        audio = np.concatenate([c.audio.detach().cpu().numpy() if hasattr(c.audio, "detach")
                                else np.asarray(c.audio) for c in chunks]) if chunks else np.zeros(1)
        sr = 24000
        audio_s = len(audio) / float(sr)
        wav = None
        if a.wav_dir:
            import soundfile as sf
            wav = os.path.join(a.wav_dir, f"kokoro_{label}.wav")
            sf.write(wav, audio, sr)
        row = {"label": label, "chars": len(text), "audio_s": round(audio_s, 3),
               "wall_s": round(synth_s, 3),
               "rtf": round(audio_s / synth_s, 2) if synth_s > 0 else None, "wav": wav}
        rows.append(row)
        print(f"  {label}: {len(text)} 字符 -> {audio_s:.2f}s 音频 / {synth_s:.3f}s 墙钟 "
              f"RTF={row['rtf']}", flush=True)

    rtfs = [r["rtf"] for r in rows if r["rtf"]]
    rep = {"schema_version": SCHEMA, "kind": "tts_kokoro_bench",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "env": env_fingerprint(a.threads),
           "args": {"kokoro_dir": os.path.basename(a.kokoro_dir.rstrip("\\/")),
                    "voice": a.voice, "smoke": a.smoke},
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
