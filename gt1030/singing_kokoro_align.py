# -*- coding: utf-8 -*-
"""singing_kokoro_align.py — Kokoro 整句合成 + 强制对齐 → 逐音唱歌（v4）。

为什么：v2 用 Piper 逐字合成，辅音保真后 ASR 覆盖 0.40；
但 Kokoro 逐字会崩（输出"有有有"），它需要上下文——所以这次先用 Kokoro 合成**整句**，
再用 faster-whisper 的 word_timestamps 做强制对齐，把整句切成 14 个字的音频段，
最后仍用 v2 的"辅音保真 + 元音延音改调"逐音重调。
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time

import librosa
import numpy as np
import soundfile as sf

OUT = r"D:\work\research\bench_results\singing"
SR = 24000
beat = 0.5

TWINKLE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]
LYRIC = list("一闪一闪亮晶晶满天都是小星星")
TIGER = [60, 62, 64, 60, 60, 62, 64, 60, 64, 65, 67, 64, 65, 67,
         67, 69, 67, 65, 64, 60, 67, 69, 67, 65, 64, 60, 60, 55, 60, 60, 55, 60]
TIGER_LYRIC = list("两只老虎两只老虎跑得快跑得快一只没有眼睛一只没有尾巴真奇怪真奇怪")
SONGS = {"twinkle_zh": (TWINKLE, LYRIC, {6, 13}), "tiger_zh": (TIGER, TIGER_LYRIC, {3, 7, 15, 23, 31})}

sys_path = r"D:\work\bitsandbytes-CPU\gt1030"
import sys
sys.path.insert(0, sys_path)
from singing_synth_v2 import sing_note, midi_hz  # noqa: E402


def kokoro_zh(voice_name="zf_xiaobei"):
    import torch
    from kokoro import KModel, KPipeline
    K = r"D:\work\textmodel\kokoro-82m"
    torch.set_num_threads(6)
    km = KModel(repo_id=None, model=os.path.join(K, "kokoro-v1_0.pth"),
                config=os.path.join(K, "config.json")).eval()
    pipe = KPipeline(lang_code="z", repo_id=None, model=km)
    v = torch.load(os.path.join(K, "voices", f"{voice_name}.pt"), map_location="cpu", weights_only=True)
    return pipe, v


def synth_phrase(pipe, voice, text):
    chunks = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(text, voice=voice, speed=1.0)]
    audio = np.concatenate(chunks).astype(np.float32)
    return audio


def char_timings(audio, sr, lyric):
    """用 faster-whisper 的 word 时间戳给每个字分配时间片。"""
    from faster_whisper import WhisperModel
    m = WhisperModel(r"D:\work\textmodel\faster-whisper-small", device="cpu",
                     compute_type="int8", cpu_threads=6)
    tmp = os.path.join(OUT, "_align_probe.wav")
    sf.write(tmp, audio, sr)
    segs, _ = m.transcribe(tmp, language="zh", beam_size=1, temperature=0.0, word_timestamps=True)
    words = []
    for s in segs:
        for w in (s.words or []):
            wd = str(w.word).strip()
            if wd:
                words.append((wd, float(w.start), float(w.end)))
    # 把识别词切成单字并按字分配时间
    chars, times = [], []
    for wd, a, b in words:
        cs = [c for c in wd if re.match(r"[\u4e00-\u9fff]", c)]
        if not cs:
            continue
        for k, c in enumerate(cs):
            chars.append(c)
            d = (b - a) / len(cs)
            times.append((a + k * d, a + (k + 1) * d))
    # 用 difflib 把识别字序列对齐到歌词
    sm = difflib.SequenceMatcher(a=lyric, b=chars)
    lyric_times = [None] * len(lyric)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                lyric_times[i1 + k] = times[j1 + k]
            # 多出来的字按邻近时间线性分摊
            for k in range(n, i2 - i1):
                if times:
                    lyric_times[i1 + k] = (times[j1][0], times[min(j2, len(times) - 1)][1])
        elif tag == "delete":
            if times:
                span = (times[max(0, j1 - 1)][1] if j1 > 0 else times[0][0],
                        times[j1][0] if j1 < len(times) else times[-1][1])
                for k in range(i1, i2):
                    frac = (k - i1 + 1) / (i2 - i1 + 1)
                    t0 = span[0] + (span[1] - span[0]) * (frac - 1 / (i2 - i1 + 1))
                    lyric_times[k] = (span[0] + (span[1] - span[0]) * frac, span[0] + (span[1] - span[0]) * frac)
    # 兜底：未分配的字按总时长均匀铺
    dur = len(audio) / sr
    for i, t in enumerate(lyric_times):
        if t is None:
            lyric_times[i] = (dur * i / len(lyric), dur * (i + 1) / len(lyric))
    # 修正单调与顺序
    fixed = []
    last = 0.0
    for a, b in lyric_times:
        a = max(a, last)
        b = max(b, a + 0.02)
        fixed.append((a, b))
        last = b
    return fixed, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="twinkle_zh", choices=["twinkle_zh", "tiger_zh"])
    ap.add_argument("--voice", default="zf_xiaobei")
    ap.add_argument("--chords", action="store_true")
    ap.add_argument("--harmony", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()
    seq, lyric, longs = SONGS[a.song]
    pipe, voice = kokoro_zh(a.voice)
    phrase = "".join(lyric)
    audio = synth_phrase(pipe, voice, phrase)
    t_tts = time.time() - t0
    print(f"  Kokoro 整句：{len(audio)/24000:.2f}s 音频 / {t_tts:.1f}s | ASR 转写见下", flush=True)

    t1 = time.time()
    times, words = char_timings(audio, 24000, lyric)
    print(f"  对齐 {t1-t0:.1f}s | 识别词 {len(words)} 个", flush=True)
    print("  前 4 个字的时段:", [(round(x[0], 2), round(x[1], 2)) for x in times[:4]], flush=True)

    # 逐音重调（v2 的辅音保真逻辑）
    segs = []
    t2 = time.time()
    notes = [(m, (2 * beat if i in longs else beat)) for i, m in enumerate(seq)]
    for (midi, dur), (a0, b0) in zip(notes, times):
        i0 = max(0, int((a0 - 0.02) * 24000))
        i1 = min(len(audio), int((b0 + 0.03) * 24000))
        seg = audio[i0:i1].astype(np.float64)
        if len(seg) < 400:
            seg = np.pad(seg, (0, 400 - len(seg)))
        segs.append(sing_note(seg, midi_hz(midi), dur))
    t_ret = time.time() - t2
    print(f"  重调 {t_ret:.1f}s", flush=True)

    y = np.zeros(int(sum(si.shape[0] for si in segs)) + SR)
    pos = 0
    for s in segs:
        y[pos:pos + len(s)] += s
        pos += len(s)
    y = y[:pos]
    if a.chords:
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
    out = os.path.join(OUT, f"v4_{a.song}_kokoro_{a.voice}.wav")
    sf.write(out, y.astype(np.float32), SR)
    rep = dict(engine="kokoro_zh_phrase_align", voice=a.voice, song=a.song, tts_s=round(t_tts, 1),
               align_s=round(t1 - t0, 1), retune_s=round(t_ret, 1),
               audio_s=round(len(y) / SR, 2), out=out,
               char_times=[(round(x[0], 2), round(x[1], 2)) for x in times])
    with open(os.path.join(OUT, f"v4_{a.song}_kokoro_{a.voice}_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"  输出 {out} | {len(y)/SR:.1f}s", flush=True)
    print("RESULT " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
