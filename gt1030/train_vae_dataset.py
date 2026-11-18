# -*- coding: utf-8 -*-
"""train_vae_dataset.py — real-image VAE reconstruction with held-out validation.

    python gt1030/train_vae_dataset.py --steps 200 --eval-every 20 \
        --train-dir D:\work\_mempool\vae_small\train \
        --val-dir   D:\work\_mempool\vae_small\val

Device: frozen SD1.5 VAE encoder+decoder, boundary LoRA, 8-bit optimizer state.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\gt1030")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import gt1030.igpu_fx as FX  # noqa: E402
from gt1030.igpu_lora import wrap_lora  # noqa: E402
from train_vae_lora import AutoEnc, build_autoencoder  # noqa: E402

SCALE = 65536.0


def load_images(folder, size):
    ts = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        im = Image.open(p).convert("RGB").resize((size, size))
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        ts.append(torch.from_numpy(arr).permute(2, 0, 1))
    if not ts:
        raise SystemExit("no images in " + folder)
    return torch.stack(ts)


def boundary_filter(nm):
    if nm.endswith("conv_in") or nm.endswith("conv_out"):
        return True
    return ".resnets.0." in nm and (nm.endswith(".conv1") or nm.endswith(".conv2"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=r"D:\work\textmodel\_sd15_combined\vae")
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--lora-scope", choices=["all", "boundary"], default="boundary")
    ap.add_argument("--arena", type=int, default=3000)
    ap.add_argument("--log-csv", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    train = load_images(a.train_dir, a.size)
    val = load_images(a.val_dir, a.size)
    print("  images: train %d / val %d · %dx%d" %
          (len(train), len(val), a.size, a.size))

    enc, dec = build_autoencoder(a.vae)
    model = AutoEnc(enc, dec)
    filt = boundary_filter if a.lora_scope == "boundary" else None
    nwrap = wrap_lora(model, r=a.lora_r, conv=True, name_filter=filt)
    train_names = sorted({
        n.rsplit(".", 1)[0] for n, _ in model.named_parameters() if ".lora_" in n
    })
    print("  LoRA layers %d · scope %s" % (nwrap, a.lora_scope))

    x0 = train[:a.batch]
    ex = FX.Executor(model, x0, state_bits=a.bits, arena_mb=a.arena,
                     freeze_base=True, trainable_names=train_names)
    ex.auto_zero_grads = True
    ex.sync_from_torch()
    print("  device arena %.0f MB" % (ex.dev._off / 1048576))

    def eval_loss():
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), a.batch):
                xb = val[i:i + a.batch]
                if len(xb) == 0:
                    continue
                y = ex.forward({"x": xb.numpy()})
                yv = ex.dev.down(y.buf, y.shape).astype(np.float32)
                tot += float(np.sum((yv - xb.numpy()) ** 2))
                cnt += yv.size
        return tot / max(cnt, 1)

    logf = open(a.log_csv, "w", buffering=1, encoding="utf-8") if a.log_csv else None
    if logf:
        logf.write("step,train_loss,val_loss,ms\n")

    order = torch.randperm(len(train))
    pos = 0
    tr_hist, va_hist, ms_hist = [], [], []
    t0 = time.time()
    for t in range(1, a.steps + 1):
        if pos + a.batch > len(train):
            order = torch.randperm(len(train))
            pos = 0
        idx = order[pos:pos + a.batch]
        pos += a.batch
        xb = train[idx]
        s = time.time()
        ex.zero_grads()
        y = ex.forward({"x": xb.numpy()})
        n = int(np.prod(y.shape))
        yv = ex.dev.down(y.buf, y.shape).astype(np.float32)
        tr_loss = float(np.mean((yv - xb.numpy()) ** 2))
        dy = (2.0 / n) * (yv - xb.numpy()) * SCALE
        dyb = ex._cached("ds.dy", (n,))
        cl.enqueue_copy(ex.dev.q, dyb, np.ascontiguousarray(dy.reshape(-1), np.float16))
        ex.backward(dyb, y.shape)
        ex.step(a.lr, t, eps=1e-8 * SCALE)
        ms = 1000 * (time.time() - s)
        tr_hist.append(tr_loss)
        ms_hist.append(ms)
        if t == 1 or t % a.eval_every == 0:
            vl = eval_loss()
            va_hist.append((t, vl))
            print("  step %4d  train %.6f  val %.6f  %.0f ms" % (t, tr_loss, vl, ms))
            if logf:
                logf.write("%d,%.8g,%.8g,%.1f\n" % (t, tr_loss, vl, ms))
    if logf:
        logf.close()
    tail = ms_hist[10:] if len(ms_hist) > 10 else ms_hist
    print("  final train %.6f · val %.6f · %.0f ms/step · %.1fs" %
          (tr_hist[-1], va_hist[-1][1], np.mean(tail), time.time() - t0))
    ok = va_hist[-1][1] < va_hist[0][1]
    print("RESULT " + ("PASS — held-out 下降" if ok else "FAIL — held-out 未下降"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
