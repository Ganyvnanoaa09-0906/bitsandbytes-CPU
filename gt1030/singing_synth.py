# -*- coding: utf-8 -*-
"""singing_synth.py — 用 TTS + pyworld 重调，让本机"唱"一首歌（CPU，无需唱歌专用模型）。

思路：
  1) 每个音符取一个音节（中文按字，英文按音节），用 Piper/Kokoro 合成单音节；
  2) pyworld 分析出 f0 / 频谱包络 / 非周期成分；
  3) 把 f0 替换成目标音高（带轻微 vibrato 与 portamento），时间轴拉伸到音符时值；
  4) 逐音符拼接（交叉淡化）+ 简单混响 + 可选和弦伴奏。

用法：
  python singing_synth.py --song twinkle_zh
  python singing_synth.py --song twinkle_en
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
HOP = 5.0          # pyworld 帧移 ms

# C 大调《小星星》C4=60
TWINKLE_MIDI = [60, 60, 67, 67, 69, 69, 67,  65, 65, 64, 64, 62, 62, 60]
ZH_LYRICS = list("一闪一闪亮晶晶满天都是小星星")

# 《两只老虎》：C 大调
TIGER_MIDI = [60, 62, 64, 60, 60, 62, 64, 60, 64, 65, 67, 64, 65, 67,
              67, 69, 67, 65, 64, 60, 67, 69, 67, 65, 64, 60, 60, 55, 60, 60, 55, 60]
TIGER_LYRICS = list("两只老虎两只老虎跑得快跑得快一只没有眼睛一只没有尾巴真奇怪真奇怪")
EN_SYLLABLES = ["twin", "kle", "twin", "kle", "lit", "tle", "star",
                "how", "i", "won", "der", "what", "you", "are"]


def piper_voice():
    from piper import PiperVoice
    return PiperVoice.load(r"D:\work\textmodel\piper-voices\zh_CN-huayan-medium.onnx")


def synth_syllable_piper(voice, text, path, length_scale=1.0):
    from piper import SynthesisConfig
    cfg = SynthesisConfig(length_scale=length_scale)
    with wave.open(path, "wb") as w:
        voice.synthesize_wav(text, w, syn_config=cfg)
    d, sr = sf.read(path, dtype="float64")
    if d.ndim > 1:
        d = d.mean(1)
    if sr != SR:
        d = librosa.resample(d, orig_sr=sr, target_sr=SR)
    return d


def kokoro_voice():
    import torch
    from kokoro import KModel, KPipeline
    K = r"D:\work\textmodel\kokoro-82m"
    torch.set_num_threads(6)
    km = KModel(repo_id=None, model=os.path.join(K, "kokoro-v1_0.pth"),
                config=os.path.join(K, "config.json")).eval()
    pipe = KPipeline(lang_code="a", repo_id=None, model=km)
    v = torch.load(os.path.join(K, "voices", "af_heart.pt"), map_location="cpu", weights_only=True)
    return pipe, v


def synth_syllable_kokoro(pipe, voice, text):
    chunks = []
    for _, _, a in pipe(text, voice=voice, speed=1.0):
        chunks.append(a.numpy() if hasattr(a, "numpy") else np.asarray(a))
    d = np.concatenate(chunks).astype(np.float64)
    if SR != 24000:
        d = librosa.resample(d, orig_sr=24000, target_sr=SR)
    return d


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def retune(d, target_hz, dur_s, vibrato=0.25, portamento_cents=60):
    """把一段语音改成目标音高、目标时长（pyworld 频谱包络保留，音色不变）。"""
    f0, sp, ap = pw.wav2world(d.astype(np.float64), SR, frame_period=HOP)
    n_src = len(f0)
    if n_src < 4:
        return np.zeros(int(dur_s * SR), np.float64)
    # 目标帧数
    n_tgt = max(8, int(round(dur_s * 1000.0 / HOP)))
    # 时间轴重采样（保持频谱包络）
    idx = np.linspace(0, n_src - 1, n_tgt)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n_src - 1)
    w = (idx - i0)[:, None]
    sp2 = (1 - w) * sp[i0] + w * sp[i1]
    ap2 = (1 - w) * ap[i0] + w * ap[i1]

    # 目标 f0：基频 + 开头 portamento + 轻微 vibrato
    t = np.arange(n_tgt) * HOP / 1000.0
    base = np.full(n_tgt, target_hz)
    # 前面 60 ms 从上一个音滑过来（这里用 ±portamento 的简单滑音近似）
    slide_n = max(1, int(0.06 * 1000 / HOP))
    start_hz = target_hz * 2 ** (-portamento_cents / 1200.0)
    base[:slide_n] = np.linspace(start_hz, target_hz, slide_n)
    # 颤音（长音才加，深度 ~0.25 半音）
    if dur_s > 0.35:
        vib = 2 ** (vibrato * np.sin(2 * np.pi * 5.5 * t + np.random.rand() * 6.28) / 12.0)
        ramp = np.clip((t - 0.15) / 0.2, 0, 1)
        base = base * (1 + (vib - 1) * ramp)
    # 清音/静音帧保持 0（否则会把噪声变调）
    voiced = f0 > 0
    vmap = voiced[i0]
    f0n = np.where(vmap, base, 0.0)

    y = pw.synthesize(f0n, sp2, ap2, SR, HOP)
    y = y[: int(dur_s * SR)]
    if len(y) < int(dur_s * SR):
        y = np.pad(y, (0, int(dur_s * SR) - len(y)))
    return y.astype(np.float64)


def add_reverb(x, sr=SR, wet=0.22, decay=0.35):
    n = int(sr * 0.9)
    ir = np.random.randn(n) * np.exp(-np.arange(n) / (sr * decay))
    ir = ir / np.max(np.abs(ir)) * 0.5
    y = np.convolve(x, ir)[: len(x) + n // 2]
    y = y[: len(x)]
    return (1 - wet) * x + wet * y


def chord_pad(midis, dur, sr=SR, amp=0.06):
    """简单的三和弦垫音（正弦 + 轻微谐波）。"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for m in midis:
        f = midi_to_hz(m)
        env = np.minimum(1.0, np.minimum(t / 0.05, (dur - t) / 0.08))
        y += amp * env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="twinkle_zh",
                    choices=["twinkle_zh", "twinkle_en", "tiger_zh"])
    ap.add_argument("--beats", type=float, default=0.5, help="每拍秒数")
    ap.add_argument("--chords", action="store_true", help="加简单和弦伴奏")
    ap.add_argument("--slow", type=float, default=1.8, help="TTS 放慢倍数（减少后续拉伸）")
    ap.add_argument("--vowel", default="", choices=["", "la", "ah"], help="用元音音节代替歌词（更像唱歌）")
    ap.add_argument("--harmony", action="store_true", help="加一条低三度的伴唱")
    ap.add_argument("--tag-extra", default="", help="输出文件名后缀（区分 voice-only / 带伴奏版）")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    beat = a.beats
    if a.song == "tiger_zh":
        notes = [(m, 1 * beat) for m in TIGER_MIDI]
        for i in (3, 7, 15, 23, 31):
            notes[i] = (TIGER_MIDI[i], 2 * beat)
    else:
        notes = [(m, 1 * beat) for m in TWINKLE_MIDI]
        notes[6] = (TWINKLE_MIDI[6], 2 * beat)
        notes[-1] = (TWINKLE_MIDI[-1], 2 * beat)

    t_all = time.time()
    if a.song.endswith("_zh"):
        voice = piper_voice()
        tmp = os.path.join(OUT, "tmp")
        os.makedirs(tmp, exist_ok=True)
        lyrics = TIGER_LYRICS if a.song == "tiger_zh" else ZH_LYRICS
        if a.vowel:
            lyrics = [{"la": "啦", "ah": "啊"}[a.vowel]] * len(lyrics)
        syl = []
        t1 = time.time()
        for i, ch in enumerate(lyrics):
            syl.append(synth_syllable_piper(voice, ch, os.path.join(tmp, f"z{i}.wav"), length_scale=a.slow))
        print(f"  合成 {len(syl)} 个音节 {time.time()-t1:.1f}s", flush=True)
    else:
        pipe, v = kokoro_voice()
        syl = []
        t1 = time.time()
        for i, s in enumerate(EN_SYLLABLES):
            syl.append(synth_syllable_kokoro(pipe, v, s))
        print(f"  合成 {len(syl)} 个音节 {time.time()-t1:.1f}s", flush=True)

    # 可选：加一条低三度伴唱（用同一批音节的音色）
    if a.harmony:
        harm = [(m - 3, d) for m, d in notes]
        segs_h = []
        for (midi, dur), d in zip(harm, syl):
            segs_h.append(retune(d, midi_to_hz(midi), dur))
    else:
        segs_h = []

    # 逐音符重调
    segs = []
    t1 = time.time()
    for (midi, dur), d in zip(notes, syl):
        segs.append(retune(d, midi_to_hz(midi), dur))
    print(f"  重调 {len(segs)} 个音符 {time.time()-t1:.1f}s", flush=True)

    # 拼接（10 ms 交叉淡化）
    total = sum(len(s) for s in segs)
    y = np.zeros(total + SR)
    pos = 0
    xf = int(0.01 * SR)
    for s in segs:
        end = pos + len(s)
        y[pos:end] += s
        if pos > 0:
            # 淡入，让前一个音的尾部自然过渡
            n = min(xf, len(s))
            y[pos:pos + n] *= np.linspace(0, 1, n)
            y[pos - n:pos] *= np.linspace(1, 0, n)[::-1] + 1e-3
        pos = end
    y = y[:pos]

    # 伴唱混入（低三度，音量 0.55）
    if a.harmony and segs_h:
        yh = np.zeros(len(y) + SR)
        pos = 0
        for s in segs_h:
            end = pos + len(s)
            yh[pos:end] += s
            pos = end
        yh = yh[:len(y)] * 0.55
        y = y + yh

    # 和弦伴奏
    if a.chords:
        chords = [(48, 55, 60), (53, 60, 65), (55, 62, 67), (48, 55, 60)]
        bar = 4 * beat
        acc = np.zeros_like(y)
        for i in range(0, len(y), int(bar * SR)):
            c = chords[(i // int(bar * SR)) % len(chords)]
            n = min(len(y) - i, int(bar * SR))
            acc[i:i + n] += chord_pad(c, n / SR)[:n]
        y = y / max(1e-9, np.max(np.abs(y))) * 0.9 + acc

    y = add_reverb(y)
    y = y / max(1e-9, np.max(np.abs(y))) * 0.95
    suffix = "_slow" if a.slow and a.slow > 1.01 else ""
    tag = a.song + ("_" + a.vowel if a.vowel else "") + a.tag_extra
    out = os.path.join(OUT, f"{tag}{suffix}.wav")
    sf.write(out, y.astype(np.float32), SR)
    total_s = len(y) / SR

    # 音准评估：逐音符测实际 f0 与目标的中位偏差（音分）
    errs = []
    pos = 0
    for (midi, dur), s in zip(notes, segs):
        f0, _, _ = pw.wav2world(s.astype(np.float64), SR, frame_period=HOP)
        v = f0[f0 > 0]
        if len(v) > 3:
            target = midi_to_hz(midi)
            cents = 1200 * np.log2(np.median(v) / target)
            errs.append(float(cents))
    rep = dict(song=a.song, notes=len(notes), audio_s=round(total_s, 2),
               wall_s=round(time.time() - t_all, 1), sr=SR, output=out,
               pitch_err_cents_median=round(float(np.median(np.abs(errs))), 1) if errs else None,
               pitch_err_cents_per_note=[round(e, 1) for e in errs])
    with open(os.path.join(OUT, f"{tag}{suffix}_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"  输出 {out} | {total_s:.1f}s 音频 | 总耗时 {rep['wall_s']}s | "
          f"音准中位偏差 {rep['pitch_err_cents_median']} 音分", flush=True)
    print("RESULT " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
