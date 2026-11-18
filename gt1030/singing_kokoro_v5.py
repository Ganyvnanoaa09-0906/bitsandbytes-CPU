# -*- coding: utf-8 -*-
"""singing_kokoro_v5.py — 两种更好的切分策略，对比 v4 的对齐效果。

A) 停顿合成：让 Kokoro 用逗号把每个字隔开（自然停顿），再用能量/VAD 找静音切成 14 段
   —— 不依赖 ASR 对齐，重复字（一闪一闪）也不会错位；
B) hotwords 对齐：faster-whisper 传歌词做 hotwords，提高转写准确率后再做 v4 式对齐。

两种都走 v2 的"辅音保真 + 元音延音改调"，最后用同一条 ASR 指标比。
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time

import librosa
import numpy as np
import soundfile as sf

OUT = r"D:\work\research\bench_results\singing"
SR = 24000
beat = 0.5
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\gt1030")
from singing_synth_v2 import sing_note, midi_hz  # noqa: E402

TWINKLE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]
LYRIC = list("一闪一闪亮晶晶满天都是小星星")
TIGER = [60, 62, 64, 60, 60, 62, 64, 60, 64, 65, 67, 64, 65, 67,
         67, 69, 67, 65, 64, 60, 67, 69, 67, 65, 64, 60, 60, 55, 60, 60, 55, 60]
TIGER_LYRIC = list("两只老虎两只老虎跑得快跑得快一只没有眼睛一只没有尾巴真奇怪真奇怪")
SONGS = {"twinkle_zh": (TWINKLE, LYRIC, {6, 13}), "tiger_zh": (TIGER, TIGER_LYRIC, {3, 7, 15, 23, 31})}


def kokoro(voice_name="zf_xiaobei"):
    import torch
    from kokoro import KModel, KPipeline
    K = r"D:\work\textmodel\kokoro-82m"
    torch.set_num_threads(6)
    km = KModel(repo_id=None, model=os.path.join(K, "kokoro-v1_0.pth"),
                config=os.path.join(K, "config.json")).eval()
    pipe = KPipeline(lang_code="z", repo_id=None, model=km)
    v = torch.load(os.path.join(K, "voices", f"{voice_name}.pt"), map_location="cpu", weights_only=True)
    return pipe, v


def synth(pipe, voice, text):
    chunks = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(text, voice=voice, speed=1.0)]
    return np.concatenate(chunks).astype(np.float32)


def split_by_silence(audio, n_expected, sr=24000):
    """按能量找静音，切成 n_expected 段；不够就合并最近的，多了就合并最短的。"""
    # 帧级 RMS
    hop = 240                      # 10 ms
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop)[0]
    thr = max(rms.min() + 1e-6, rms.mean() * 0.25)
    voiced = rms > thr
    # 找连续 voiced 区间
    segs = []
    i = 0
    n = len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            if (j - i) * hop > 0.04 * sr:   # 至少 40 ms
                segs.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    if not segs:
        d = len(audio) / n_expected
        return [(int(k * d), int((k + 1) * d)) for k in range(n_expected)]
    # 调整数量：多了合并间隔最短的相邻段；少了按最长段等分
    while len(segs) > n_expected:
        gaps = [segs[k + 1][0] - segs[k][1] for k in range(len(segs) - 1)]
        k = int(np.argmin(gaps))
        segs[k] = (segs[k][0], segs[k + 1][1])
        del segs[k + 1]
    while len(segs) < n_expected:
        lens = [b - a for a, b in segs]
        k = int(np.argmax(lens))
        a, b = segs[k]
        mid = (a + b) // 2
        segs[k] = (a, mid)
        segs.insert(k + 1, (mid, b))
    return segs


def char_times_whisper(audio, lyric, hotwords=None):
    from faster_whisper import WhisperModel
    m = WhisperModel(r"D:\work\textmodel\faster-whisper-small", device="cpu",
                     compute_type="int8", cpu_threads=6)
    tmp = os.path.join(OUT, "_align_probe2.wav")
    sf.write(tmp, audio, 24000)
    kwargs = dict(language="zh", beam_size=1, temperature=0.0, word_timestamps=True)
    if hotwords:
        kwargs["hotwords"] = hotwords
    segs, _ = m.transcribe(tmp, **kwargs)
    words = []
    for s in segs:
        for w in (s.words or []):
            wd = str(w.word).strip()
            if wd:
                words.append((wd, float(w.start), float(w.end)))
    chars, times = [], []
    for wd, a, b in words:
        cs = [c for c in wd if re.match(r"[\u4e00-\u9fff]", c)]
        for k, c in enumerate(cs):
            chars.append(c)
            d = (b - a) / max(1, len(cs))
            times.append((a + k * d, a + (k + 1) * d))
    sm = difflib.SequenceMatcher(a=lyric, b=chars)
    out = [None] * len(lyric)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(min(i2 - i1, j2 - j1)):
                out[i1 + k] = times[j1 + k]
    dur = len(audio) / 24000
    for i, t in enumerate(out):
        if t is None:
            out[i] = (dur * i / len(lyric), dur * (i + 1) / len(lyric))
    fixed, last = [], 0.0
    for a, b in out:
        a = max(a, last); b = max(b, a + 0.02)
        fixed.append((a, b)); last = b
    return fixed


def render_from_times(song, audio, times, chords=False):
    seq, lyric, longs = SONGS[song]
    notes = [(m, (2 * beat if i in longs else beat)) for i, m in enumerate(seq)]
    segs = []
    for (midi, dur), (a0, b0) in zip(notes, times):
        i0 = max(0, int((a0 - 0.015) * SR))
        i1 = min(len(audio), int((b0 + 0.03) * SR))
        seg = audio[i0:i1].astype(np.float64)
        if len(seg) < 400:
            seg = np.pad(seg, (0, 400 - len(seg)))
        segs.append(sing_note(seg, midi_hz(midi), dur))
    y = np.zeros(int(sum(s.shape[0] for s in segs)) + SR)
    pos = 0
    for s in segs:
        y[pos:pos + len(s)] += s
        pos += len(s)
    y = y[:pos]
    if chords:
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
    return (y / max(1e-9, np.max(np.abs(y))) * 0.95).astype(np.float32)


def asr_cover(path):
    from faster_whisper import WhisperModel
    m = WhisperModel(r"D:\work\textmodel\faster-whisper-small", device="cpu",
                     compute_type="int8", cpu_threads=6)
    ref = "".join(SONGS["twinkle_zh" if "twinkle" in os.path.basename(path) else "tiger_zh"][1])
    want = set(re.findall(r"[\u4e00-\u9fff]", ref))
    segs, _ = m.transcribe(path, language="zh", beam_size=1, temperature=0.0)
    text = "".join(s.text for s in segs)
    got = set(re.findall(r"[\u4e00-\u9fff]", text))
    return round(len(got & want) / max(1, len(want)), 3), text.strip()[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="twinkle_zh", choices=list(SONGS))
    ap.add_argument("--voice", default="zf_xiaobei")
    ap.add_argument("--strategy", default="both", choices=["pause", "hotwords", "both"])
    ap.add_argument("--chords", action="store_true")
    a = ap.parse_args()
    seq, lyric, _ = SONGS[a.song]
    pipe, voice = kokoro(a.voice)
    rows = []

    if a.strategy in ("pause", "both"):
        t0 = time.time()
        spaced = "，".join(lyric)
        audio = synth(pipe, voice, spaced)
        t_tts = time.time() - t0
        segs = split_by_silence(audio, len(lyric))
        times = [(s / SR, e / SR) for s, e in segs]
        y = render_from_times(a.song, audio, times, chords=a.chords)
        p = os.path.join(OUT, f"v5p_{a.song}_{a.voice}.wav")
        sf.write(p, y, SR)
        cov, text = asr_cover(p)
        rows.append(dict(strategy="pause", tts_s=round(t_tts, 1), segments=len(segs),
                         cov=cov, text=text, out=p))
        print(f"  pause: 段数 {len(segs)} | 覆盖 {cov} | {text[:50]}", flush=True)

    if a.strategy in ("hotwords", "both"):
        t0 = time.time()
        audio = synth(pipe, voice, "".join(lyric))
        t_tts = time.time() - t0
        times = char_times_whisper(audio, lyric, hotwords="".join(lyric))
        y = render_from_times(a.song, audio, times, chords=a.chords)
        p = os.path.join(OUT, f"v5h_{a.song}_{a.voice}.wav")
        sf.write(p, y, SR)
        cov, text = asr_cover(p)
        rows.append(dict(strategy="hotwords", tts_s=round(t_tts, 1), cov=cov, text=text, out=p))
        print(f"  hotwords: 覆盖 {cov} | {text[:50]}", flush=True)

    with open(os.path.join(OUT, f"v5_{a.song}_{a.voice}_compare.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
