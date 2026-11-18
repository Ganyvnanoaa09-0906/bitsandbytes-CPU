# -*- coding: utf-8 -*-
"""singing_eval.py — 评估"唱"出来的东西：可懂度（ASR）+ 音准曲线图。"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pyworld as pw
import soundfile as sf

OUT = r"D:\work\research\bench_results\singing"
SING = {
    "twinkle_zh": dict(lang="zh", lyric="一闪一闪亮晶晶满天都是小星星"),
    "twinkle_en": dict(lang="en", lyric="twinkle twinkle little star how i wonder what you are"),
    "tiger_zh": dict(lang="zh", lyric="两只老虎两只老虎跑得快跑得快一只没有眼睛一只没有尾巴真奇怪真奇怪"),
}


def asr_eval():
    from faster_whisper import WhisperModel
    m = WhisperModel(r"D:\work\textmodel\faster-whisper-small", device="cpu",
                     compute_type="int8", cpu_threads=6)
    rows = {}
    for tag, meta in SING.items():
        p = os.path.join(OUT, f"{tag}.wav")
        if not os.path.exists(p):
            continue
        segs, info = m.transcribe(p, language=meta["lang"], beam_size=1, temperature=0.0)
        text = "".join(s.text for s in segs) if meta["lang"] == "zh" else " ".join(s.text for s in segs)
        ref = meta["lyric"]
        # 字符级覆盖率（中文）/ 词级（英文）
        if meta["lang"] == "zh":
            gotchars = set(re.findall(r"[\u4e00-\u9fff]", text))
            refchars = set(re.findall(r"[\u4e00-\u9fff]", ref))
            cover = len(gotchars & refchars) / max(1, len(refchars))
        else:
            gw = set(re.findall(r"[a-z]+", text.lower()))
            rw = set(re.findall(r"[a-z]+", ref.lower()))
            cover = len(gw & rw) / max(1, len(rw))
        rows[tag] = dict(text=text.strip()[:200], char_coverage=round(cover, 3))
        print(f"  {tag}: ASR「{text.strip()[:80]}」| 覆盖 {cover:.2f}", flush=True)
    return rows


def f0_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  matplotlib 缺失，跳过 F0 图:", repr(e)[:60])
        return None
    fig, axes = plt.subplots(len(SING), 1, figsize=(10, 2.4 * len(SING)), sharex=False)
    if len(SING) == 1:
        axes = [axes]
    for ax, (tag, meta) in zip(axes, SING.items()):
        p = os.path.join(OUT, f"{tag}.wav")
        if not os.path.exists(p):
            continue
        d, sr = sf.read(p, dtype="float64")
        if d.ndim > 1:
            d = d.mean(1)
        f0, t = pw.harvest(d, sr, frame_period=10.0)
        f0[f0 <= 0] = np.nan
        ax.plot(t, f0, lw=1.2, color="#2b6cb0")
        ax.set_title(f"{tag}  (F0)", fontsize=9)
        ax.set_ylabel("Hz", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT, "f0_curves.png")
    plt.savefig(out, dpi=110)
    plt.close()
    print("  [save] f0_curves.png", flush=True)
    return out


def main():
    rows = asr_eval()
    plot = f0_plot()
    rep = dict(asr=rows, plot=plot)
    with open(os.path.join(OUT, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
