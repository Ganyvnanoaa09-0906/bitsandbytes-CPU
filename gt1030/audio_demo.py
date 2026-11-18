# -*- coding: utf-8 -*-
"""audio_demo.py — 生成中英文 demo 音频（Piper 中文 + Kokoro 英文），并写 A.16.26 报告。"""
from __future__ import annotations

import json
import os
import time
import wave

import numpy as np
import soundfile as sf

OUT = r"D:\work\research\bench_results\audio_lab"
PIPER_ZH = r"D:\work\textmodel\piper-voices\zh_CN-huayan-medium.onnx"
KOKORO = r"D:\work\textmodel\kokoro-82m"

ZH = ("你好，这里是本机上的语音合成演示。所有模型都在没有独立显卡的普通笔记本上运行，"
      "不需要联网。中文用的是 Piper 的音色，速度快，几乎瞬间就能生成。"
      "如果你听到这段话，说明这台机器的音频链路已经跑通了。")
EN = ("Hello, this is a text to speech demo running entirely on a laptop without a graphics card. "
      "The English voice you are hearing is Kokoro, an eighty two million parameter model. "
      "It runs about twice as fast as real time on this machine. "
      "If you can understand this sentence, the whole audio pipeline works.")


def main():
    t0 = time.time()
    rows = []

    # 1) 中文（Piper）
    from piper import PiperVoice
    v = PiperVoice.load(PIPER_ZH)
    zh_path = os.path.join(OUT, "demo_zh_piper.wav")
    t1 = time.time()
    with wave.open(zh_path, "wb") as w:
        v.synthesize_wav(ZH, w)
    dt = time.time() - t1
    d, sr = sf.read(zh_path)
    rows.append(dict(lang="zh", engine="piper_zh_huayan", audio_s=round(len(d) / sr, 2),
                     wall_s=round(dt, 2), rtf=round(dt / (len(d) / sr), 3), path=zh_path))
    print(f"  中文 Piper: {len(d)/sr:.2f}s 音频 / {dt:.2f}s 合成（RTF {dt/(len(d)/sr):.3f}）", flush=True)

    # 2) 英文（Kokoro）
    import torch
    from kokoro import KModel, KPipeline
    torch.set_num_threads(6)
    km = KModel(repo_id=None, model=os.path.join(KOKORO, "kokoro-v1_0.pth"),
                config=os.path.join(KOKORO, "config.json")).eval()
    pipe = KPipeline(lang_code="a", repo_id=None, model=km)
    voice = torch.load(os.path.join(KOKORO, "voices", "af_heart.pt"), map_location="cpu", weights_only=True)
    t1 = time.time()
    chunks = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(EN, voice=voice, speed=1.0)]
    audio = np.concatenate(chunks)
    dt = time.time() - t1
    en_path = os.path.join(OUT, "demo_en_kokoro.wav")
    sf.write(en_path, audio, 24000)
    rows.append(dict(lang="en", engine="kokoro_af_heart", audio_s=round(len(audio) / 24000, 2),
                     wall_s=round(dt, 2), rtf=round(dt / (len(audio) / 24000), 3), path=en_path))
    print(f"  英文 Kokoro: {len(audio)/24000:.2f}s 音频 / {dt:.2f}s 合成（RTF {dt/(len(audio)/24000):.3f}）", flush=True)

    # 3) 两段拼接（中间 0.6 s 静音）
    zh_a, zh_sr = sf.read(zh_path)
    gap = np.zeros(int(0.6 * 24000), np.float32)
    from scipy.signal import resample_poly
    zh24 = resample_poly(zh_a, 24000, zh_sr).astype(np.float32)
    allwav = np.concatenate([zh24, gap, audio.astype(np.float32)])
    all_path = os.path.join(OUT, "demo_zh_en.wav")
    sf.write(all_path, allwav, 24000)
    print(f"  拼接 demo: {len(allwav)/24000:.2f}s -> {all_path}", flush=True)

    with open(os.path.join(OUT, "demo_report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
