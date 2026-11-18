# -*- coding: utf-8 -*-
"""log_train.py — **外部**逐步骤记录（不改 train_vae_lora.py），用于长跑诊断。

    python gt1030/log_train.py --steps 200 --csv gt1030/_log200.csv
    python gt1030/log_train.py --steps 1000 --bits 8 --lr 2e-4 --no-image --csv ...

输出 CSV：step, loss, ms, adapter_l2（每 --probe-every 步测一次）
跑完打印：末值、前 50 步之后的方差、是否出现尖峰（|Δloss| > 5×中位数）。

**等价性验证**：同样参数下末步 loss 应与 `train_vae_lora.py` 一致（±1e-3），
不一致就说明本封装抄错了训练循环，结果作废。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\gt1030")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import gt1030.igpu_fx as FX  # noqa: E402
from gt1030.igpu_lora import wrap_lora  # noqa: E402

SCALE = 65536.0


def load_image(path, size=64):
    from PIL import Image

    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(im, np.float32) / 127.5 - 1.0
    return np.transpose(arr, (2, 0, 1))[None].repeat(2, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--scope", default="boundary")
    ap.add_argument("--image", default=r"D:\work\_mempool\sample.png")
    ap.add_argument("--csv", default=r"gt1030\_log.csv")
    ap.add_argument("--probe-every", type=int, default=10)
    a = ap.parse_args()

    from train_vae_lora import AutoEnc, build_autoencoder

    torch.manual_seed(0)
    enc, dec = build_autoencoder(r"D:\work\textmodel\_sd15_combined\vae")
    model = AutoEnc(enc, dec)
    wrap_lora(model, r=a.lora_r, conv=True)
    names = [n for n, _ in model.named_parameters() if ".lora_" in n]
    prefixes = sorted({n.rsplit(".", 1)[0] for n in names})

    x_np = load_image(a.image)
    x = torch.from_numpy(x_np)
    ex = FX.Executor(model, x, state_bits=a.bits, arena_mb=2500,
                     freeze_base=True, trainable_names=prefixes)
    ex.auto_zero_grads = True
    ex.sync_from_torch()
    adapter_params = [p for n, p in ex.params.items() if ".lora_" in n]
    print("  实验：%d 步 · lr=%.0e · bits=%d · r=%d · scope=%s · adapter %d 个"
          % (a.steps, a.lr, a.bits, a.lora_r, a.scope, len(adapter_params)))

    rows, t_start = [], time.time()
    for t in range(1, a.steps + 1):
        s = time.time()
        ex.zero_grads()
        y = ex.forward({"x": x_np})
        n = int(np.prod(y.shape))
        yv = ex.dev.down(y.buf, y.shape).astype(np.float32)
        loss = float(np.mean((yv - x_np) ** 2))
        dy = (2.0 / n) * (yv - x_np) * SCALE
        dyb = ex._cached("log.dy", (n,))
        cl.enqueue_copy(ex.dev.q, dyb, np.ascontiguousarray(dy.reshape(-1), np.float16))
        ex.backward(dyb, y.shape)
        ex.step(a.lr, t, eps=1e-8 * SCALE)
        ms = 1000 * (time.time() - s)
        l2 = ""
        if t % a.probe_every == 0 or t == 1:
            acc = 0.0
            for p in adapter_params:
                w = ex.dev.down(p.buf, p.shape).astype(np.float32)
                acc += float(np.sum(w * w))
            l2 = "%.6f" % np.sqrt(acc)
        rows.append((t, loss, ms, l2))

    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "loss", "ms", "adapter_l2"])
        w.writerows(rows)

    losses = np.array([r[1] for r in rows], np.float64)
    tail = losses[49:] if len(losses) > 50 else losses
    diffs = np.abs(np.diff(losses))
    med = float(np.median(diffs)) if diffs.size else 0.0
    spikes = int(np.sum(diffs > 5 * med)) if med > 0 else 0
    print("  末步 loss %.6f · 首步 %.6f · 下降 %.1f%%"
          % (losses[-1], losses[0], 100 * (1 - losses[-1] / losses[0])))
    print("  前 50 步后 loss：均值 %.6f · 方差 %.3e · 标准差 %.6f"
          % (tail.mean(), tail.var(), tail.std()))
    print("  尖峰(|Δloss|>5×中位差 %.3e)：%d 次" % (med, spikes))
    print("  步时中位 %.0f ms · 总 %.1f s · CSV %s"
          % (float(np.median([r[2] for r in rows])), time.time() - t_start, a.csv))
    print("RESULT " + ("PASS" if np.isfinite(losses).all() else "FAIL-NaN"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
