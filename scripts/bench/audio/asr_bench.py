# -*- coding: utf-8 -*-
"""asr_bench.py — CPU 上的 faster-whisper 推理实测（RTF / 内存 / 可选 WER）。

依赖（不入库，自行安装）：
    pip install faster-whisper soundfile psutil
权重（不入库）：
    --model 指向 CTranslate2 格式的 faster-whisper 目录（含 model.bin / config.json / tokenizer.json）

用法：
    python scripts/bench/audio/asr_bench.py --model D:\\models\\faster-whisper-small \\
        --audio-dir audio/ --out out/asr.json --compute-type int8
    python ... --smoke            # 只跑第一个 wav，用于快速验证环境

输出：JSON（schema v1）+ 每个文件的转写文本。RTF = 音频时长 / 墙钟耗时（>1 表示快于实时）。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import sys
import time
from _common import check_threads, load_json, write_json

SCHEMA = 1


def env_fingerprint(threads: int) -> dict:
    fp = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": threads,
    }
    try:
        import psutil

        fp["cpu_count"] = psutil.cpu_count(logical=True)
        fp["rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 3)
    except Exception:
        pass
    try:
        import faster_whisper

        fp["faster_whisper"] = getattr(faster_whisper, "__version__", "?")
    except Exception:
        pass
    return fp


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def wer(ref: str, hyp: str) -> float:
    """词错误率（Levenshtein 在词序列上）。"""
    r, h = norm(ref), norm(hyp)
    if not r:
        return float("nan")
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="faster-whisper 模型目录或名字")
    ap.add_argument("--audio-dir", required=True, help="含 *.wav 的目录")
    ap.add_argument("--out", default="", help="JSON 输出路径（空=只打印）")
    ap.add_argument(
        "--refs", default="", help="可选：{文件名: 参考文本} 的 JSON，用于算 WER"
    )
    ap.add_argument(
        "--compute-type", default="int8", choices=["int8", "int8_float32", "float32"]
    )
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--language", default="en")
    ap.add_argument("--smoke", action="store_true", help="只跑第一个文件（禁用大文件）")
    a = ap.parse_args()
    a.threads = check_threads(a.threads)

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(f"[skip] 缺少依赖 faster-whisper: {e}", file=sys.stderr)
        return 2
    if not os.path.isdir(a.model):
        # 只接受本地 CTranslate2 目录：传模型名会让 faster-whisper 联网下载，违背"权重不入库/不下载"约定
        print(
            f"[skip] 模型目录不存在（本脚本只接受本地目录，避免联网下载）: {a.model}",
            file=sys.stderr,
        )
        return 2
    if not os.path.isdir(a.audio_dir):
        print(f"[skip] 音频目录不存在: {a.audio_dir}", file=sys.stderr)
        return 2

    random.seed(0)
    wavs = sorted(f for f in os.listdir(a.audio_dir) if f.lower().endswith(".wav"))
    if not wavs:
        print("[skip] 目录里没有 .wav", file=sys.stderr)
        return 2
    if a.smoke:
        wavs = wavs[:1]
    refs = load_json(a.refs, "refs") if a.refs else {}

    t0 = time.time()
    model = WhisperModel(
        a.model, device="cpu", compute_type=a.compute_type, cpu_threads=a.threads
    )
    load_s = time.time() - t0

    rows = []
    for w in wavs:
        path = os.path.join(a.audio_dir, w)
        t1 = time.time()
        try:
            segments, info = model.transcribe(path, language=a.language, beam_size=1)
            text = " ".join(s.text.strip() for s in segments).strip()
            dt = time.time() - t1
        except Exception as e:  # 坏 wav / 解码失败：记录该文件，不炸整轮
            dt = time.time() - t1
            rows.append(
                {
                    "file": w,
                    "audio_s": None,
                    "wall_s": round(dt, 3),
                    "rtf": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            print(
                f"  {w}: [skip] 无法解码/转写: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue
        dur = float(getattr(info, "duration", 0.0)) or 0.0
        row = {
            "file": w,
            "audio_s": round(dur, 3),
            "wall_s": round(dt, 3),
            "rtf": round(dur / dt, 3) if dt > 0 and dur > 0 else None,
            "text": text,
        }
        if w in refs:
            row["wer"] = round(wer(refs[w], text), 4)
        rows.append(row)
        print(
            f"  {w}: {dur:.2f}s 音频 / {dt:.2f}s 墙钟  RTF={row['rtf']}  {text[:60]!r}",
            flush=True,
        )

    if rows and all("error" in r for r in rows):
        print("[skip] 所有音频都无法解码/转写", file=sys.stderr)
        return 2
    rtf = [r["rtf"] for r in rows if r["rtf"]]
    rep = {
        "schema_version": SCHEMA,
        "kind": "asr_bench",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": env_fingerprint(a.threads),
        "args": {
            "model": os.path.basename(a.model.rstrip("\\/")),
            "compute_type": a.compute_type,
            "language": a.language,
            "smoke": a.smoke,
            "n_files": len(rows),
        },
        "load_s": round(load_s, 3),
        "results": rows,
        "summary": {
            "mean_rtf": round(sum(rtf) / len(rtf), 3) if rtf else None,
            "mean_wer": round(
                sum(r.get("wer", 0) for r in rows if "wer" in r)
                / max(1, sum(1 for r in rows if "wer" in r)),
                4,
            )
            if any("wer" in r for r in rows)
            else None,
        },
    }
    print("  " + json.dumps(rep["summary"], ensure_ascii=False))
    if a.out:
        write_json(a.out, rep)
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
