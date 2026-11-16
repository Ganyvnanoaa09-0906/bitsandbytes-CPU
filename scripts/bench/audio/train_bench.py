# -*- coding: utf-8 -*-
"""train_bench.py — CPU 上音频模型"训练"的单步成本实测（三档规模）。

三档：
  1) tiny_cnn   —— 唤醒词级别（~0.1M 参数），真跑 fwd+bwd；
  2) vits_proxy —— TTS 声学模型量级（~20M 参数，用同规模 MLP/Conv 代理，不依赖 torchaudio 版本）；
  3) whisper_small_proxy —— ASR 微调量级（~185M 参数，Transformer 编码器代理）。

目的：给出"一步多久 / N 步要多少小时"的可外推数字，而不是训练精度。

依赖（不入库）：pip install torch psutil
用法：
    python scripts/bench/audio/train_bench.py --out out/train.json
    python ... --smoke          # 每档只跑 2 步

输出 JSON（schema v1）。所有形状与参数量都写进结果，便于核对。
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from _common import check_threads, write_json

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
        import torch

        fp["torch"] = torch.__version__
    except Exception:
        pass
    return fp


def build_cases():
    import torch
    import torch.nn as nn

    class TinyCNN(nn.Module):
        """KWS 量级：64 mel x 64 帧 -> 4 类。"""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 96, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(96, 4),
            )

        def forward(self, x):
            return self.net(x)

    class VitsProxy(nn.Module):
        """20M 量级代理：Conv 前端 + 3 层 1024 宽 MLP。"""

        def __init__(self):
            super().__init__()
            self.front = nn.Sequential(nn.Conv1d(80, 512, 3, padding=1), nn.ReLU())
            self.body = nn.Sequential(
                nn.Linear(512, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 512),
            )

        def forward(self, x):  # (B, 80, T)
            h = self.front(x).mean(-1)
            return self.body(h)

    class WhisperProxy(nn.Module):
        """185M 量级代理：6 层 1024 宽 Transformer 编码器。"""

        def __init__(self, d=1024, layers=12, nhead=8):
            super().__init__()
            self.inp = nn.Linear(80, d)
            layer = nn.TransformerEncoderLayer(
                d, nhead, dim_feedforward=4 * d, batch_first=True, dropout=0.0
            )
            self.enc = nn.TransformerEncoder(layer, layers)
            self.head = nn.Linear(d, 80)

        def forward(self, x):  # (B, T, 80)
            return self.head(self.enc(self.inp(x)))

    def nparams(m):
        return sum(p.numel() for p in m.parameters())

    cases = []
    m = TinyCNN()
    cases.append(
        (
            "tiny_cnn",
            m,
            torch.randn(8, 1, 64, 64),
            torch.randint(0, 4, (8,)),
            nparams(m),
        )
    )
    m = VitsProxy()
    cases.append(
        ("vits_proxy", m, torch.randn(4, 80, 128), torch.randn(4, 512), nparams(m))
    )
    m = WhisperProxy()
    cases.append(
        (
            "whisper_small_proxy",
            m,
            torch.randn(2, 64, 80),
            torch.randn(2, 64, 80),
            nparams(m),
        )
    )
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="JSON 输出路径")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument(
        "--steps", type=int, default=10, help="每档计时的步数（不含 1 步预热）"
    )
    ap.add_argument("--smoke", action="store_true", help="每档只跑 2 步")
    a = ap.parse_args()
    a.threads = check_threads(a.threads)

    try:
        import torch
        # psutil 只用于环境指纹；缺失时降级，不要让"可选依赖"变成硬依赖
    except Exception as e:
        print(f"[skip] 缺少依赖 torch: {e}", file=sys.stderr)
        return 2
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    steps = 2 if a.smoke else max(1, a.steps)

    rows = []
    for name, model, x, y, npar in build_cases():
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lossf = (
            torch.nn.CrossEntropyLoss() if name == "tiny_cnn" else torch.nn.MSELoss()
        )
        # 预热 1 步
        opt.zero_grad()
        out = model(x)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
        t0 = time.time()
        for _ in range(steps):
            opt.zero_grad()
            out = model(x)
            loss = lossf(out, y)
            loss.backward()
            opt.step()
        dt = (time.time() - t0) / steps
        rows.append(
            {
                "case": name,
                "params_m": round(npar / 1e6, 3),
                "step_s": round(dt, 4),
                "steps": steps,
                "steps_per_s": round(1.0 / dt, 3) if dt > 0 else None,
                "hours_for_10k_steps": round(dt * 1e4 / 3600, 2),
            }
        )
        print(
            f"  {name}: {npar / 1e6:.2f}M 参数  单步 {dt * 1000:.1f} ms  "
            f"(1 万步 ≈ {dt * 1e4 / 3600:.2f} h)",
            flush=True,
        )

    rep = {
        "schema_version": SCHEMA,
        "kind": "train_bench",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": env_fingerprint(a.threads),
        "args": {"smoke": a.smoke, "steps": steps},
        "results": rows,
        "summary": {
            "steps_per_case": steps,
            "step_s": {r["case"]: r["step_s"] for r in rows},
            "hours_for_10k_steps": {r["case"]: r["hours_for_10k_steps"] for r in rows},
        },
        "notes": ["proxy 模型只用于量级外推，不代表实际模型结构/精度"],
    }
    if a.out:
        write_json(a.out, rep)
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
