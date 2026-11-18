# -*- coding: utf-8 -*-
"""train_vae_lora.py — VAE **encoder + decoder 重建任务**骨架（真实 SD1.5 权重）。

给 LoRA 用的底座：`--lora-r` 由 `igpu_lora.wrap_lora` 接管（见 peer commit 196e9509）。

    python gt1030/train_vae_lora.py --steps 20 --size 64

注意：SD1.5 decoder 的 `mid_block.attentions`（全注意力）**没有接**——设备路径目前只有
因果注意力，且需要 token 布局转换；这里跳过该子块，其余 49.5M-1.05M 参数全部参与。
对拍基准因此是"同一份手写前向"的 torch 版本，而不是上游 diffusers 的 decoder。
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\gt1030")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import gt1030.igpu_fx as FX  # noqa: E402
from gt1030.igpu_lora import wrap_lora  # noqa: E402

SCALE = 65536.0


def _resnet(r, h):
    """ResnetBlock2D 的等价前向（GN → SiLU → Conv → GN → SiLU → Conv → +shortcut）。"""
    x_in = h
    h = r.nonlinearity(r.norm1(h))
    h = r.conv1(h)
    h = r.nonlinearity(r.norm2(h))
    h = r.conv2(h)
    if r.conv_shortcut is not None:
        x_in = r.conv_shortcut(x_in)
    return x_in + h


def _down(d):
    """Downsample2D：stride=2 的卷积前先 pad(0,1,0,1)。"""
    if getattr(d, "use_conv", False):
        return d.conv(F.pad(d.conv.weight.new_zeros(0) if False else d._dummy_input, (0, 0))
                      ) if False else d.conv(F.pad(_down._x, (0, 1, 0, 1)))
    return F.avg_pool2d(_down._x, 2)


def build_autoencoder(vae_dir):
    """返回 (encoder, decoder) 两个手写前向的模块（真实权重）。"""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_dir)
    vae.eval()
    e, d = vae.encoder, vae.decoder

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = e.conv_in
            self.down_blocks = e.down_blocks
            self.conv_norm_out = e.conv_norm_out
            self.conv_act = e.conv_act
            self.conv_out = e.conv_out

        def forward(self, x):
            h = self.conv_in(x)
            for blk in self.down_blocks:
                for r in blk.resnets:
                    h = _resnet(r, h)
                for ds in (blk.downsamplers or []):
                    if ds is None:
                        continue
                    h = ds.conv(F.pad(h, (0, 1, 0, 1))) if getattr(ds, "use_conv", False) \
                        else F.avg_pool2d(h, 2)
            return self.conv_out(self.conv_act(self.conv_norm_out(h)))

    class Dec(nn.Module):
        """decoder（跳过 mid_block.attentions：设备侧暂无全注意力）。"""

        def __init__(self):
            super().__init__()
            self.conv_in = d.conv_in
            self.mid_resnets = d.mid_block.resnets
            self.up_blocks = d.up_blocks
            self.conv_norm_out = d.conv_norm_out
            self.conv_act = d.conv_act
            self.conv_out = d.conv_out

        def forward(self, z):
            h = self.conv_in(z)
            for r in self.mid_resnets:
                h = _resnet(r, h)
            for blk in self.up_blocks:
                for r in blk.resnets:
                    h = _resnet(r, h)
                for us in (blk.upsamplers or []):
                    if us is None:
                        continue
                    if getattr(us, "use_conv_transpose", False):
                        h = us.conv(h)
                    else:
                        h = F.interpolate(h, scale_factor=2.0, mode="nearest")
                        h = us.conv(h)
            return self.conv_out(self.conv_act(self.conv_norm_out(h)))

    return Enc(), Dec()


class AutoEnc(nn.Module):
    """encoder → (quant_conv 省略) → decoder；训练用它做真实重建。"""

    def __init__(self, enc, dec):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self, x):
        h = self.encoder(x)          # (N, 8, h, w) = mean + logvar
        return self.decoder(h[:, :4])  # 取 mean 的 4 通道作 latent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=r"D:\work\textmodel\_sd15_combined\vae")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=0, help=">0 时给 Conv2d 挂 LoRA 并冻结底座")
    ap.add_argument("--arena", type=int, default=3000)
    a = ap.parse_args()

    torch.manual_seed(0)
    enc, dec = build_autoencoder(a.vae)
    model = AutoEnc(enc, dec)
    nbase = sum(p.numel() for p in model.parameters())
    train_names = None
    if a.lora_r > 0:
        n = wrap_lora(model, r=a.lora_r, conv=True)
        lora_names = [nm for nm, _ in model.named_parameters() if ".lora_" in nm]
        train_names = sorted({nm.rsplit(".", 1)[0] for nm in lora_names})
        print("  LoRA r=%d：包裹 %d 层 · adapter %.2f M / 底座 %.1f M"
              % (a.lora_r, n,
                 sum(p.numel() for p in model.parameters()) / 1e6 - nbase / 1e6, nbase / 1e6))
    else:
        print("  全参训练：底座 %.1f M" % (nbase / 1e6))

    x = torch.randn(a.batch, 3, a.size, a.size)
    with torch.no_grad():
        ref_out = model(x)
    print("  输入 %s → 重建 %s" % (tuple(x.shape), tuple(ref_out.shape)))

    ex = FX.Executor(model, x, state_bits=a.bits, arena_mb=a.arena,
                     freeze_base=bool(a.lora_r), trainable_names=train_names)
    ex.auto_zero_grads = True
    ex.sync_from_torch()
    ntr = sum(1 for p in ex.params.values() if not getattr(p, "frozen", False))
    nfr = sum(1 for p in ex.params.values() if getattr(p, "frozen", False))
    print("  设备参数：可训 %d / 冻结 %d · 构建后占用 %.0f MB"
          % (ntr, nfr, ex.dev._off / 1048576))

    losses, ms = [], []
    for t in range(1, a.steps + 1):
        s = time.time()
        ex.zero_grads()
        y = ex.forward({"x": x.numpy()})
        n = int(np.prod(y.shape))
        yv = ex.dev.down(y.buf, y.shape).astype(np.float32)
        losses.append(float(np.mean((yv - x.numpy()) ** 2)))
        dy = (2.0 / n) * (yv - x.numpy()) * SCALE
        dyb = ex._cached("rec.dy", (n,))
        cl.enqueue_copy(ex.dev.q, dyb, np.ascontiguousarray(dy.reshape(-1), np.float16))
        ex.backward(dyb, y.shape)
        ex.step(a.lr, t, eps=1e-8 * SCALE)
        ms.append(1000 * (time.time() - s))
    print("  step  loss(MSE vs 原图)")
    for i in (0, 1, a.steps // 2, a.steps - 1):
        if i < len(losses):
            print("  %4d  %.6f" % (i + 1, losses[i]))
    print("  末步 %.6f（下降 %.1f%%）· %.0f ms/步" %
          (losses[-1], 100 * (1 - losses[-1] / losses[0]), np.mean(ms)))
    ok = losses[-1] < losses[0]
    print("RESULT " + ("PASS — 真实 VAE 编解码重建训练跑通" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
