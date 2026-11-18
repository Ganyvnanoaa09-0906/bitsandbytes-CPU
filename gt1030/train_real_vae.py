# -*- coding: utf-8 -*-
"""train_real_vae.py — 用**真实 SD1.5 VAE 编码器权重**在核显上训练（dev13 第一个真实模型）。

不直接 trace diffusers 的 ResnetBlock2D（内部有控制流，symbolic_trace 会报错），
而是用它的算子手写等价前向：GN → SiLU → Conv → GN → SiLU → Conv → +shortcut。
这样图里只有 Conv2d / GroupNorm / SiLU / add——全部已支持。

    python gt1030/train_real_vae.py --steps 120
    python gt1030/train_real_vae.py --steps 120 --bits 8 --size 64
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import gt1030.igpu_fx as FX  # noqa: E402

SCALE = 65536.0


def build_encoder(vae_dir, size):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_dir)
    vae.eval()
    e = vae.encoder

    class Enc(nn.Module):
        """手写的 encoder 前向：避开 diffusers 内部的控制流。"""

        def __init__(self):
            super().__init__()
            self.conv_in = e.conv_in
            self.down_blocks = e.down_blocks
            self.conv_norm_out = e.conv_norm_out
            self.conv_act = e.conv_act
            self.conv_out = e.conv_out

        @staticmethod
        def _resnet(r, h):
            x_in = h
            h = r.nonlinearity(r.norm1(h))
            h = r.conv1(h)
            h = r.nonlinearity(r.norm2(h))
            h = r.conv2(h)
            if r.conv_shortcut is not None:
                x_in = r.conv_shortcut(x_in)
            return x_in + h

        def forward(self, x):
            h = self.conv_in(x)
            for blk in self.down_blocks:
                for r in blk.resnets:
                    h = self._resnet(r, h)
                if getattr(blk, "downsamplers", None) is not None:
                    for d in blk.downsamplers:
                        if d is None:
                            continue
                        if getattr(d, "use_conv", False):
                            # diffusers 的 Downsample2D：stride=2 时先 pad(0,1,0,1) 再卷积
                            if type(d).__name__.startswith("Downsample2D"):
                                h = d.conv(F.pad(h, (0, 1, 0, 1)))
                            else:
                                h = d.conv(h)
                        else:
                            h = F.avg_pool2d(h, 2)
            h = self.conv_norm_out(h)
            h = self.conv_act(h)
            return self.conv_out(h)

    return Enc()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=r"D:\work\textmodel\_sd15_combined\vae")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--arena", type=int, default=3000)
    a = ap.parse_args()

    torch.manual_seed(0)
    enc = build_encoder(a.vae, a.size)
    nparam = sum(p.numel() for p in enc.parameters())
    print("  真实 SD1.5 VAE 编码器：%.1f M 参数 · 输入 %d×3×%d×%d"
          % (nparam / 1e6, a.batch, a.size, a.size))

    x = torch.randn(a.batch, 3, a.size, a.size)
    with torch.no_grad():
        tgt = enc(x) + torch.randn_like(enc(x)) * 0.1   # 目标：编码输出 + 噪声（可训的回归任务）

    # ---------- 核显 ----------
    ex = FX.Executor(enc, x, state_bits=a.bits, arena_mb=a.arena)
    ex.auto_zero_grads = True
    ex.sync_from_torch()
    td = ex.dev.up(tgt.numpy().astype(np.float16))
    dl, ms = [], []
    t0 = time.time()
    for t in range(1, a.steps + 1):
        s = time.time()
        ex.zero_grads()
        y = ex.forward({"x": x.numpy()})
        n = int(np.prod(y.shape))
        dy_np = (2.0 / n) * (ex.dev.down(y.buf, y.shape).astype(np.float32)
                             - tgt.numpy()) * SCALE
        dl.append(float(np.mean((ex.dev.down(y.buf, y.shape).astype(np.float32)
                                 - tgt.numpy()) ** 2)))
        dyb = ex._cached("train.dy", (n,))
        cl.enqueue_copy(ex.dev.q, dyb, np.ascontiguousarray(dy_np.reshape(-1), np.float16))
        ex.backward(dyb, y.shape)
        ex.step(a.lr, t, eps=1e-8 * SCALE)
        ms.append(1000 * (time.time() - s))
    dev_wall = time.time() - t0

    # ---------- torch 参考 ----------
    ref = build_encoder(a.vae, a.size)
    ref.load_state_dict(enc.state_dict())
    opt = torch.optim.Adam(ref.parameters(), lr=a.lr, eps=1e-8)
    tl = []
    t0 = time.time()
    for t in range(a.steps):
        opt.zero_grad()
        loss = ((ref(x) - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        tl.append(float(loss.detach()))
    cpu_wall = time.time() - t0

    print("  step   device      torch")
    for i in (0, 1, 4, a.steps // 2, a.steps - 1):
        if i < a.steps:
            print("  %4d  %10.6f  %10.6f" % (i + 1, dl[i], tl[i]))
    print("  末步相对差 %.2e" % (abs(dl[-1] - tl[-1]) / max(abs(tl[-1]), 1e-12)))
    print("  步时: device %.1f ms · torch(CPU) %.1f ms · 加速 %.2fx"
          % (np.mean(ms), 1000 * cpu_wall / a.steps,
             (cpu_wall / a.steps) / (np.mean(ms) / 1000)))
    import gt1030.igpu_fx as _FX
    nbuf = len(getattr(ex, "_node_bufs", {}))
    print("  设备缓冲 %d 个 · arena 用 %.0f MB" % (nbuf, ex.dev._off / 1048576))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
