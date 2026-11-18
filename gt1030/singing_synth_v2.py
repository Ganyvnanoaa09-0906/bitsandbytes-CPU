# -*- coding: utf-8 -*-
"""singing_synth_v2.py — 改进版"唱歌"：辅音不拉伸，只在元音上延音+改调。

v1 的问题：整段音节（含辅音）被时间拉伸 + 改调 → 辅音/声调被抹平，ASR 字符覆盖只有 0.10。
v2 的做法：
  1) WORLD 分析整个音节，找到第一个浊音帧 i0（辅音结束、元音开始）；
  2) 头部 [0, i0) 用原始 f0 合成（辅音原样保留，不改时长）；
  3) 尾部（元音）时间轴重采样填满音符剩余时值，f0 换成目标音高（+滑入/颤音）；
  4) 头尾拼接（5 ms 交叉淡化）。
"""
from __future__ import annotations

import argparse
import json
import os
import time
import wave

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

OUT = r"D:\work\research\bench_results\singing"
SR = 24000
HOP = 5.0

TWINKLE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]
ZH = list("一闪一闪亮晶晶满天都是小星星")

TIGER = [60, 62, 64, 60, 60, 62, 64, 60, 64, 65, 67, 64, 65, 67,
         67, 69, 67, 65, 64, 60, 67, 69, 67, 65, 64, 60, 60, 55, 60, 60, 55, 60]
TIGER_ZH = list("两只老虎两只老虎跑得快跑得快一只没有眼睛一只没有尾巴真奇怪真奇怪")
SONGS = {"twinkle_zh": (TWINKLE, ZH), "tiger_zh": (TIGER, TIGER_ZH)}


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def kokoro_zh():
    import torch
    from kokoro import KModel, KPipeline
    K = r"D:\work\textmodel\kokoro-82m"
    torch.set_num_threads(6)
    km = KModel(repo_id=None, model=os.path.join(K, "kokoro-v1_0.pth"),
                config=os.path.join(K, "config.json")).eval()
    pipe = KPipeline(lang_code="z", repo_id=None, model=km)
    v = torch.load(os.path.join(K, "voices", "zf_xiaobei.pt"), map_location="cpu", weights_only=True)
    return pipe, v


def synth_kokoro_zh(pipe, voice, text, path):
    import torch
    chunks = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(text, voice=voice, speed=1.0)]
    d = np.concatenate(chunks).astype(np.float64)
    if len(d) == 0:
        return np.zeros(int(0.2 * SR))
    if SR != 24000:
        d = librosa.resample(d, orig_sr=24000, target_sr=SR)
    return d


def piper_voice():
    from piper import PiperVoice
    return PiperVoice.load(r"D:\work\textmodel\piper-voices\zh_CN-huayan-medium.onnx")


def synth_piper(voice, text, path, length_scale=1.0):
    from piper import SynthesisConfig
    with wave.open(path, "wb") as w:
        voice.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=length_scale))
    d, sr = sf.read(path, dtype="float64")
    if d.ndim > 1:
        d = d.mean(1)
    if sr != SR:
        d = librosa.resample(d, orig_sr=sr, target_sr=SR)
    return d


def sing_note(d, target_hz, dur_s, vibrato=0.25, use_head=True):
    """辅音保真 + 元音延音改调。"""
    f0, sp, ap = pw.wav2world(d.astype(np.float64), SR, frame_period=HOP)
    n = len(f0)
    if n < 4:
        return np.zeros(int(dur_s * SR))
    voiced = f0 > 0
    idx_v = np.where(voiced)[0]
    n_tgt = max(8, int(round(dur_s * 1000 / HOP)))

    if not use_head or len(idx_v) == 0:
        # 退化：整段重采样 + 改调
        head_n = 0
        i0 = 0
    else:
        i0 = max(2, int(idx_v[0]))
        head_n = i0                      # 头部帧数（辅音）
    n_tail_src = max(1, n - i0)
    n_tail_tgt = max(1, n_tgt - head_n)

    # --- 头部：原始 f0，原时长 ---
    if head_n > 0:
        yh = pw.synthesize(f0[:head_n].copy(), sp[:head_n], ap[:head_n], SR, HOP)
    else:
        yh = np.zeros(0)

    # --- 尾部：时间重采样 + 目标 f0 ---
    idx = np.linspace(0, n_tail_src - 1, n_tail_tgt)
    a = np.floor(idx).astype(int) + i0
    b = np.minimum(a + 1, n - 1)
    w = (idx - np.floor(idx))[:, None]
    sp2 = (1 - w) * sp[a] + w * sp[b]
    ap2 = (1 - w) * ap[a] + w * ap[b]
    t = np.arange(n_tail_tgt) * HOP / 1000.0
    base = np.full(n_tail_tgt, target_hz)
    slide = max(1, min(int(0.05 * 1000 / HOP), max(1, n_tail_tgt // 2)))
    base[:slide] = np.linspace(target_hz * 2 ** (-60 / 1200), target_hz, slide)
    if dur_s > 0.35:
        vib = 2 ** (vibrato * np.sin(2 * np.pi * 5.5 * t) / 12.0)
        ramp = np.clip((t - 0.12) / 0.2, 0, 1)
        base = base * (1 + (vib - 1) * ramp)
    vm = (f0[a] > 0)
    f0t = np.where(vm, base, 0.0)
    yt = pw.synthesize(f0t, sp2, ap2, SR, HOP)

    y = np.concatenate([yh, yt])
    need = int(dur_s * SR)
    if len(y) < need:
        y = np.pad(y, (0, need - len(y)))
    y = y[:need]
    # 头尾 5 ms 交叉淡化（避免爆音）
    xf = min(80, len(yh) // 2 if len(yh) > 0 else 0)
    if xf > 4 and len(yt) > xf and len(y) >= len(yh) + xf and len(yh) > xf:
        y_c = y.copy()
        ramp = np.linspace(0, 1, xf)
        y_c[len(yh) - xf:len(yh)] = (yh[-xf:] * (1 - ramp) + yt[:xf] * ramp)
        y = y_c
    return y


def render(song, slow, chords=True, harmony=True, tag="", output_dir=OUT, engine="piper"):
    os.makedirs(output_dir, exist_ok=True)
    beat = 0.5
    seq, lyric = SONGS[song]
    notes = [(m, beat) for m in seq]
    if song == "twinkle_zh":
        notes[6] = (seq[6], 2 * beat)
        notes[13] = (seq[13], 2 * beat)
    else:
        for i in (3, 7, 15, 23, 31):
            notes[i] = (seq[i], 2 * beat)
    tmp = os.path.join(output_dir, "tmp2")
    os.makedirs(tmp, exist_ok=True)
    syl = []
    t0 = time.time()
    if engine == "kokoro":
        pipe, kv = kokoro_zh()
        for i, ch in enumerate(lyric):
            syl.append(synth_kokoro_zh(pipe, kv, ch, os.path.join(tmp, f"k{i}.wav")))
    else:
        voice = piper_voice()
        for i, ch in enumerate(lyric):
            syl.append(synth_piper(voice, ch, os.path.join(tmp, f"v{i}.wav"), length_scale=slow))
    t_syn = time.time() - t0
    t0 = time.time()
    segs = [sing_note(d, midi_hz(m), dur) for (m, dur), d in zip(notes, syl)]
    t_ret = time.time() - t0
    y = np.zeros(int(sum(n[1] for n in notes) * SR) + SR)
    pos = 0
    for s in segs:
        y[pos:pos + len(s)] += s
        pos += len(s)
    y = y[:pos]
    if chords:
        # 简单和弦垫音
        ch = [(48, 55, 60), (53, 60, 65), (55, 62, 67), (48, 55, 60)]
        bar = int(4 * beat * SR)
        acc = np.zeros(len(y))
        for i in range(0, len(y), bar):
            c = ch[(i // bar) % 4]
            n = min(len(y) - i, bar)
            t = np.arange(n) / SR
            pad = np.zeros(n)
            for m in c:
                f = midi_hz(m)
                env = np.minimum(1, np.minimum(t / 0.05, (n / SR - t) / 0.08))
                pad += 0.05 * env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))
            acc[i:i + n] += pad
        y = y / max(1e-9, np.max(np.abs(y))) * 0.9 + acc
    y = y / max(1e-9, np.max(np.abs(y))) * 0.95
    out = os.path.join(output_dir, f"v2_{song}{tag}.wav") if engine == "piper" else os.path.join(output_dir, f"v3_{song}{tag}.wav")
    sf.write(out, y.astype(np.float32), SR)
    rep = dict(song=song, slow=slow, tts_s=round(t_syn, 1), retune_s=round(t_ret, 1),
               total_s=round(t_syn + t_ret, 1), audio_s=round(len(y) / SR, 2), out=out)
    with open(os.path.join(output_dir, ("v2" if engine == "piper" else "v3") + f"_{song}{tag}_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"  {out} | 音频 {len(y)/SR:.1f}s | TTS {t_syn:.1f}s + 重调 {t_ret:.1f}s", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", type=float, default=1.5)
    ap.add_argument("--no-chords", action="store_true")
    ap.add_argument("--song", default="twinkle_zh", choices=["twinkle_zh", "tiger_zh"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--engine", choices=["piper", "kokoro"], default="piper")
    a = ap.parse_args()
    render(a.song, a.slow, chords=not a.no_chords, tag=a.tag, engine=a.engine)


if __name__ == "__main__":
    main()
