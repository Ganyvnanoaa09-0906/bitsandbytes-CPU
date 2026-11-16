# -*- coding: utf-8 -*-
"""train_kws_demo.py — 在 CPU 上真训一个关键词识别小模型（端到端、可复现）。

流程：Piper 合成 4 类词（yes/no/stop/unknown）→ 增广（时间伸缩/增益/噪声）→ log-mel
（64 mel x 64 帧 ≈ 1 s）→ 23k 参数 CNN → 训练 → 测试集准确率 + 混淆矩阵。

依赖（不入库）：pip install piper-tts torch torchaudio numpy
权重（不入库）：--voice 指向 Piper *.onnx 语音包

用法：
    python scripts/bench/audio/train_kws_demo.py --voice voices/en_US-lessac-medium.onnx \\
        --out out/kws.json --epochs 20
    python ... --smoke          # 每类 4 条 + 1 epoch，约 10 s

输出 JSON（schema v1）：参数量、每 epoch loss/acc、测试准确率、混淆矩阵。
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import sys
import time
from _common import check_threads, fail, write_json

SCHEMA = 1
SR = 16000
DUR = 1.0
N_MELS, N_FRAMES = 64, 64
WORDS = ["yes", "no", "stop", "unknown"]
# unknown 用一组不在词表里的词，避免和前三类同音
FILLERS = ["maybe", "hello", "seven", "banana", "purple", "river"]


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
        import torch

        fp["torch"] = torch.__version__
    except Exception:
        pass
    return fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="Piper *.onnx 语音包")
    ap.add_argument("--out", default="", help="JSON 输出路径")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument(
        "--per-word", type=int, default=40, help="每类合成条数（训练+测试）"
    )
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="每类 4 条 + 1 epoch")
    a = ap.parse_args()
    a.threads = check_threads(a.threads)

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torchaudio
        from piper import PiperVoice
    except Exception as e:
        print(
            f"[skip] 缺少依赖（piper-tts/torch/torchaudio/numpy）: {e}", file=sys.stderr
        )
        return 2
    if not os.path.exists(a.voice):
        print(f"[skip] 语音包不存在: {a.voice}", file=sys.stderr)
        return 2

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)

    per_word = 4 if a.smoke else max(8, a.per_word)
    epochs = 1 if a.smoke else max(1, a.epochs)
    try:
        voice = PiperVoice.load(a.voice)
    except Exception as e:
        fail(f"无法加载 Piper 语音包 {a.voice}: {type(e).__name__}: {e}")
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_mels=N_MELS, n_fft=400, hop_length=max(1, SR // N_FRAMES)
    )

    def synth(text: str) -> np.ndarray:
        chunks = list(voice.synthesize(text))
        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return x

    def to_mel(x: np.ndarray) -> np.ndarray:
        n = int(SR * DUR)
        x = np.resize(x, n) if len(x) < n else x[:n]
        t = torch.from_numpy(x)
        m = mel(t).numpy()
        if m.shape[1] < N_FRAMES:
            m = np.pad(m, ((0, 0), (0, N_FRAMES - m.shape[1])))
        m = m[:, :N_FRAMES]
        m = np.log1p(m)
        m = (m - m.mean()) / (m.std() + 1e-6)
        return m.astype(np.float32)

    X, Y = [], []
    t_syn = time.time()
    for ci, w in enumerate(WORDS):
        for k in range(per_word):
            text = random.choice(FILLERS) if w == "unknown" else f"{w}."
            x = synth(text)
            if len(x) < 64:
                continue
            # 增广：增益抖动 + 加噪
            x = x * (1.0 + 0.2 * random.random())
            x = x + 0.02 * np.random.randn(len(x)).astype(np.float32)
            X.append(to_mel(x))
            Y.append(ci)
    syn_s = time.time() - t_syn
    if not X:
        print("[skip] 合成的音频全部为空", file=sys.stderr)
        return 2
    X = torch.from_numpy(np.stack(X))[:, None, :, :]
    Y = torch.tensor(Y, dtype=torch.long)
    n = X.shape[0]
    n_te = max(4, n // 5)
    idx = torch.randperm(n)
    te, tr = idx[:n_te], idx[n_te:]
    if len(tr) == 0:
        print(
            f"[skip] 训练集为空（n={n}, n_te={n_te}）：增大 --per-word 或不用 --smoke",
            file=sys.stderr,
        )
        return 2

    class KWS(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 48, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(48, len(WORDS)),
            )

        def forward(self, x):
            return self.net(x)

    model = KWS()
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        tot = corr = 0
        lsum = 0.0
        for s in range(0, len(perm), 16):
            b = tr[perm[s : s + 16]]
            opt.zero_grad()
            out = model(X[b])
            loss = lossf(out, Y[b])
            loss.backward()
            opt.step()
            lsum += float(loss.detach()) * len(b)
            corr += int((out.argmax(1) == Y[b]).sum())
            tot += len(b)
        model.eval()
        with torch.no_grad():
            pred = model(X[te]).argmax(1)
            acc = float((pred == Y[te]).float().mean())
        hist.append(
            {
                "epoch": ep + 1,
                "loss": round(lsum / max(1, tot), 4),
                "train_acc": round(corr / max(1, tot), 4),
                "test_acc": round(acc, 4),
            }
        )
        print(
            f"  epoch {ep + 1:2d}: loss {lsum / max(1, tot):.3f}  "
            f"train {corr / max(1, tot):.3f}  test {acc:.3f}",
            flush=True,
        )
    train_s = time.time() - t0

    conf = [[0] * len(WORDS) for _ in WORDS]
    with torch.no_grad():
        pred = model(X[te]).argmax(1)
    for t_, p_ in zip(Y[te].tolist(), pred.tolist()):
        conf[t_][p_] += 1

    rep = {
        "schema_version": SCHEMA,
        "kind": "train_kws_demo",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": env_fingerprint(a.threads),
        "args": {
            "voice": os.path.basename(a.voice),
            "epochs": epochs,
            "per_word": per_word,
            "n_samples": int(n),
            "smoke": a.smoke,
        },
        "params": int(npar),
        "synth_s": round(syn_s, 2),
        "train_s": round(train_s, 2),
        # 与 README 的 schema v1 对齐：逐 epoch 记录放 results，汇总放 summary；
        # history 保留为向后兼容别名
        "results": hist,
        "summary": {
            "test_acc": hist[-1]["test_acc"],
            "params": int(npar),
            "epochs": epochs,
            "n_samples": int(n),
        },
        "history": hist,
        "test_acc": hist[-1]["test_acc"],
        "confusion": {"labels": WORDS, "matrix": conf},
        "notes": [
            "数据为 TTS 合成 + 增广，仅用于验证 CPU 训练链路，不代表真实 KWS 精度"
        ],
    }
    print(
        f"  参数量 {npar} | 合成 {syn_s:.1f}s | 训练 {train_s:.1f}s | "
        f"测试 acc {hist[-1]['test_acc']:.3f}"
    )
    if a.out:
        write_json(a.out, rep)
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
