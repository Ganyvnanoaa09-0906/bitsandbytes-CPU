# -*- coding: utf-8 -*-
"""igpu_unet_dev.py — 设备侧 UNet `ResnetBlock2D`（含 temb），逐中间张量对拍。

结构（与 diffusers 逐位一致，见 igpu_unet_blocks.resnet_forward）：
    h = GN1(x) → SiLU → conv1 → **+ temb_proj(SiLU(temb)) 按通道广播**
      → GN2 → SiLU → conv2 → (shortcut(x) + h) / output_scale_factor

对拍**四个中间张量**（peer 要求）：conv1 后、temb 加后、conv2 后、最终 out。
只对最终 out 有可能是两错抵消。

    python gt1030/igpu_unet_dev.py --check-resnet
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.igpu_conv import Conv2d, K as CVK  # noqa: E402
from gt1030.igpu_gn import GroupNorm  # noqa: E402
from gt1030.igpu_graph import _kernels as GK  # noqa: E402

SILU = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void silu_fwd(__global const half* x, __global half* y, long n) {
    long i = get_global_id(0);
    if (i >= n) return;
    float v = (float)x[i];
    y[i] = (half)(v / (1.f + exp(-v)));
}
"""
_S = {}


def SK(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _S:
        prog = cl.Program(dev.ctx, SILU).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _S[key] = cl.Kernel(prog, "silu_fwd")
    return _S[key]


def silu(dev, src, dst, n):
    SK(dev)(dev.q, (int(n),), None, src, dst, np.int64(n))
    return dst


def _nchw(dev, y_nhwc, N, C, H, W, out):
    CVK(dev)["nhwc2nchw"](dev.q, (N * C * H * W,), None, y_nhwc, out,
                          np.int32(N), np.int32(C), np.int32(H), np.int32(W))
    return out


def _add(dev, a, b, out, n):
    GK(dev)["add_h"](dev.q, (int(n),), None, a, b, out, np.int64(n))
    return out


def dev_resnet(dev, r, x_np, temb_np, groups=32):
    """x_np (N,C,H,W) fp16/fp32、temb_np (N,D)。返回 (out_np, 中间张量 dict)。"""
    import torch

    N, C, H, W = x_np.shape
    C2 = r.conv1.out_channels
    eps = float(r.norm1.eps)

    x = dev.up(np.asarray(x_np, np.float16))
    gn1 = GroupNorm(dev, N, C, H, W, groups, eps=eps)
    cl.enqueue_copy(dev.q, gn1.gamma, np.ascontiguousarray(r.norm1.weight.detach().numpy().astype(np.float16)))
    cl.enqueue_copy(dev.q, gn1.beta, np.ascontiguousarray(r.norm1.bias.detach().numpy().astype(np.float16)))
    y1 = dev.buf(max(4096, N * C * H * W * 2))
    gn1.forward(x, y1)
    s1 = dev.buf(max(4096, N * C * H * W * 2))
    silu(dev, y1, s1, N * C * H * W)

    c1 = Conv2d(dev, N, C, H, W, C2, 3, 3, 1, 1, 1, 1, bias=True)
    w1 = dev.up(np.ascontiguousarray(r.conv1.weight.detach().numpy().astype(np.float16)))
    b1 = dev.up(np.ascontiguousarray(r.conv1.bias.detach().numpy().astype(np.float16)))
    y2 = c1.forward(s1, w1, b1)
    c1n = dev.buf(max(4096, N * C2 * H * W * 2))
    _nchw(dev, y2, N, C2, H, W, c1n)          # ← 「conv1 后」（NCHW，给对拍用）

    # temb 投影：先 SiLU，再 Linear（D → C2），按通道广播加
    tp = r.time_emb_proj
    ts = np.asarray(temb_np, np.float32)
    ts = ts / (1.0 + np.exp(-ts))
    tproj = ts @ tp.weight.detach().numpy().T + tp.bias.detach().numpy()
    tb = dev.up(np.ascontiguousarray(tproj.astype(np.float16)))
    # ★ bias_add_oc 的索引是 m*OC+o（**NHWC**），所以要在 conv1 的 NHWC 输出上加，
    #   不能在转成 NCHW 之后加——那会按错误的内存顺序广播。
    y3n = dev.buf(max(4096, N * C2 * H * W * 2))
    cl.enqueue_copy(dev.q, y3n, y2)
    CVK(dev)["bias_add_oc"](dev.q, (N * H * W, C2), None, y3n, tb,
                            np.int32(N * H * W), np.int32(C2))
    y3 = dev.buf(max(4096, N * C2 * H * W * 2))
    _nchw(dev, y3n, N, C2, H, W, y3)          # ← 「temb 加后」（NCHW）

    gn2 = GroupNorm(dev, N, C2, H, W, groups, eps=eps)
    cl.enqueue_copy(dev.q, gn2.gamma, np.ascontiguousarray(r.norm2.weight.detach().numpy().astype(np.float16)))
    cl.enqueue_copy(dev.q, gn2.beta, np.ascontiguousarray(r.norm2.bias.detach().numpy().astype(np.float16)))
    y4 = dev.buf(max(4096, N * C2 * H * W * 2))
    gn2.forward(y3, y4)
    s2 = dev.buf(max(4096, N * C2 * H * W * 2))
    silu(dev, y4, s2, N * C2 * H * W)

    c2c = Conv2d(dev, N, C2, H, W, C2, 3, 3, 1, 1, 1, 1, bias=True)
    w2 = dev.up(np.ascontiguousarray(r.conv2.weight.detach().numpy().astype(np.float16)))
    b2 = dev.up(np.ascontiguousarray(r.conv2.bias.detach().numpy().astype(np.float16)))
    y5 = c2c.forward(s2, w2, b2)
    c2n = dev.buf(max(4096, N * C2 * H * W * 2))
    _nchw(dev, y5, N, C2, H, W, c2n)          # ← 「conv2 后」

    # shortcut：identity 或 1×1 conv
    if getattr(r, "conv_shortcut", None) is not None:
        sc = Conv2d(dev, N, C, H, W, C2, 1, 1, 1, 1, 0, 0, bias=True)
        ws = dev.up(np.ascontiguousarray(r.conv_shortcut.weight.detach().numpy().astype(np.float16)))
        bs = dev.up(np.ascontiguousarray(r.conv_shortcut.bias.detach().numpy().astype(np.float16)))
        sy = sc.forward(x, ws, bs)
        scn = dev.buf(max(4096, N * C2 * H * W * 2))
        _nchw(dev, sy, N, C2, H, W, scn)
    else:
        scn = x
    tot = dev.buf(max(4096, N * C2 * H * W * 2))
    _add(dev, scn, c2n, tot, N * C2 * H * W)
    dev.q.finish()
    scale = float(getattr(r, "output_scale_factor", 1.0))
    out = dev.down(tot, (N, C2, H, W)).astype(np.float32) / scale
    inter = {
        "conv1": dev.down(c1n, (N, C2, H, W)).astype(np.float32),
        "temb_add": dev.down(y3, (N, C2, H, W)).astype(np.float32),
        "conv2": dev.down(c2n, (N, C2, H, W)).astype(np.float32),
    }
    return out, inter


def check_resnet(seed=0, N=1, C=320, H=16, W=16, block=(0, 0)):
    import torch
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    u = UNet2DConditionModel.from_pretrained(r"D:\work\textmodel\_sd15_combined\unet")
    u.eval()
    r = u.down_blocks[block[0]].resnets[block[1]]
    x = torch.randn(N, C, H, W)
    temb = torch.randn(N, u.config.block_out_channels[-1])

    # torch 侧逐中间量
    with torch.no_grad():
        h = r.norm1(x); h = r.nonlinearity(h); h = r.conv1(h)
        ref_c1 = h.clone()
        t = r.nonlinearity(temb); t = r.time_emb_proj(t)[:, :, None, None]
        h = h + t
        ref_ta = h.clone()
        h = r.norm2(h); h = r.nonlinearity(h); h = r.conv2(h)
        ref_c2 = h.clone()
        sc = r.conv_shortcut(x) if r.conv_shortcut is not None else x
        ref_out = (sc + h) / r.output_scale_factor

    dev = CTX.DevCtx(arena_mb=1200, name="resnet")
    dev.activate()
    out, inter = dev_resnet(dev, r, x.numpy(), temb.numpy())
    rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    r1 = rel(inter["conv1"], ref_c1.numpy())
    r2 = rel(inter["temb_add"], ref_ta.numpy())
    r3 = rel(inter["conv2"], ref_c2.numpy())
    r4 = rel(out, ref_out.numpy())
    print("  resnet 设备侧: conv1 %.2e · +temb %.2e · conv2 %.2e · out %.2e"
          % (r1, r2, r3, r4))
    return max(r1, r2, r3, r4) < 2e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-resnet", action="store_true")
    a = ap.parse_args()
    if not a.check_resnet:
        print(__doc__)
        return 0
    ok = check_resnet()
    ok &= check_resnet(seed=1, C=320, H=8, W=8, block=(1, 0))
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
