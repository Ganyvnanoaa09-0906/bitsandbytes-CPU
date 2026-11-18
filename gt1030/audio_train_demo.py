# -*- coding: utf-8 -*-
"""audio_train_demo.py — CPU 上真训一个"关键词识别"小模型（4 类：yes / no / stop / unknown）。

数据用 Piper 合成（本机离线、18× 实时），加时间伸缩/增益/噪声增广；
特征用 torchaudio 的 log-mel（64 带 × 64 帧 ≈ 1 s）；
模型是与 A.16.25 同一量级的 23k 参数 CNN。
输出：训练曲线、测试集准确率、混淆矩阵、模型文件。
"""
from __future__ import annotations

import json
import os
import random
import time
import wave

import numpy as np
import torch
import torch.nn as nn
import torchaudio

OUT = r"D:\work\research\bench_results\audio_train"
VOICE = r"D:\work\textmodel\piper-voices\en_US-lessac-medium.onnx"
SR = 16000
DUR = 1.0
N_MELS, N_FRAMES = 64, 64
torch.set_num_threads(6)

WORDS = {
    "yes": ["yes", "yes please", "yes sir"],
    "no": ["no", "no thanks", "no way"],
    "stop": ["stop", "stop now", "please stop"],
}
UNKNOWN = ["open the window", "what time is it", "the weather is nice today", "call me later",
           "hello there", "good morning", "turn on the light", "hello world"]


def synth(voice, text, path):
    with wave.open(path, "wb") as w:
        voice.synthesize_wav(text, w)


def load_wav(path, sr=SR):
    import soundfile as sf
    data, sr0 = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(1)
    wav = torch.from_numpy(data).unsqueeze(0)
    if sr0 != sr:
        wav = torchaudio.functional.resample(wav, sr0, sr)
    return wav


def to_feat(wav, rng, augment=False):
    # 随机增益与噪声（训练时再加语速抖动）
    if augment:
        speed = rng.uniform(0.85, 1.15)
        wav = torchaudio.functional.resample(wav, int(16000 * speed), 16000)
    wav = wav * float(rng.uniform(0.6, 1.4))
    wav = wav + torch.randn_like(wav) * float(rng.uniform(0.0, 0.01))
    n = int(DUR * SR)
    if wav.shape[1] < n:
        wav = torch.nn.functional.pad(wav, (0, n - wav.shape[1]))
    else:
        start = 0 if not augment else rng.randint(0, max(0, wav.shape[1] - n))
        wav = wav[:, start:start + n]
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_mels=N_MELS, n_fft=400, hop_length=SR // N_FRAMES)
    logmel = torch.log(mel(wav) + 1e-6)
    logmel = torch.nn.functional.interpolate(logmel.unsqueeze(0), size=(N_MELS, N_FRAMES),
                                             mode="bilinear", align_corners=False).squeeze(0)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-5)
    return logmel.squeeze(0)   # (64, 64)，模型里再补 batch/chan 维


class KWS(nn.Module):
    def __init__(self, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def main():
    from piper import PiperVoice
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(0)
    t0 = time.time()
    voice = PiperVoice.load(VOICE)

    # 1) 合成数据（4 类 × 每类 40 条）
    raw = {k: [] for k in list(WORDS) + ["unknown"]}
    tmp = os.path.join(OUT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    idx = 0
    for cls, texts in WORDS.items():
        for i in range(40):
            txt = rng.choice(texts)
            p = os.path.join(tmp, f"{cls}_{i}.wav")
            synth(voice, txt, p)
            raw[cls].append(load_wav(p))
    for i in range(120):
        txt = rng.choice(UNKNOWN)
        p = os.path.join(tmp, f"unknown_{i}.wav")
        synth(voice, txt, p)
        raw["unknown"].append(load_wav(p))
    print(f"  合成 {sum(len(v) for v in raw.values())} 条音频 | {time.time()-t0:.1f}s", flush=True)

    # 2) 特征
    classes = ["yes", "no", "stop", "unknown"]
    X, y = [], []
    for ci, cls in enumerate(classes):
        for wav in raw[cls]:
            X.append(to_feat(wav, rng, augment=True))
            y.append(ci)
    X = torch.stack(X)
    y = torch.tensor(y)
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    X, y = X[perm], y[perm]
    n_tr = int(n * 0.8)
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    print(f"  特征 {tuple(Xtr.shape)} 训练 / {tuple(Xte.shape)} 测试", flush=True)

    # 3) 训练
    model = KWS(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    bs = 16
    hist = []
    t1 = time.time()
    for ep in range(60):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(idx)
            correct += int((model(Xtr[idx]).argmax(1) == ytr[idx]).sum())
            tot += len(idx)
        model.eval()
        with torch.no_grad():
            acc = float((model(Xte).argmax(1) == yte).float().mean())
        hist.append(dict(epoch=ep + 1, loss=round(loss_sum / tot, 4),
                         train_acc=round(correct / tot, 4), test_acc=round(acc, 4)))
        if (ep + 1) % 12 == 0:
            print(f"  epoch {ep+1}: loss {loss_sum/tot:.3f} | train {correct/tot:.3f} | test {acc:.3f}", flush=True)
    train_s = time.time() - t1

    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1)
    cm = [[0] * len(classes) for _ in classes]
    for t, p in zip(yte.tolist(), pred.tolist()):
        cm[t][p] += 1
    acc = float((pred == yte).float().mean())
    torch.save(model.state_dict(), os.path.join(OUT, "kws_cpu.pth"))
    rep = dict(n=dict(train=len(Xtr), test=len(Xte)), params=sum(p.numel() for p in model.parameters()),
               train_s=round(train_s, 1), final_test_acc=round(acc, 4),
               confusion=dict(classes=classes, matrix=cm), history=hist,
               synth_s=round(t1 - t0, 1))
    with open(os.path.join(OUT, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"  训练 {train_s:.1f}s | 测试准确率 {acc:.3f} | 参数 {rep['params']}", flush=True)
    print("RESULT " + json.dumps({k: rep[k] for k in ("params", "train_s", "final_test_acc", "confusion")},
                                 ensure_ascii=False))


if __name__ == "__main__":
    main()
