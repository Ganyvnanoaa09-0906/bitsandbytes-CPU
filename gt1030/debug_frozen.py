# -*- coding: utf-8 -*-
"""debug_frozen.py — 定位"冻结底座"路径上与 torch 的偏差：逐层比对 dy 与 conv_out 梯度。

    python gt1030/debug_frozen.py
"""
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\gt1030")
import numpy as np
import pyopencl as cl
import torch

import gt1030.igpu_fx as FX
from train_real_vae import build_encoder

SCALE = 65536.0


def main() -> int:
    torch.manual_seed(0)
    enc = build_encoder(r"D:\work\textmodel\_sd15_combined\vae", 64)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        tgt = enc(x) + torch.randn_like(enc(x)) * 0.1

    ex = FX.Executor(enc, x, state_bits=32, arena_mb=2500,
                     freeze_base=True, trainable_names=["conv_out"])
    ex.auto_zero_grads = True
    ex.sync_from_torch()
    ex.zero_grads()
    y = ex.forward({"x": x.numpy()})
    n = int(np.prod(y.shape))
    yv = ex.dev.down(y.buf, y.shape).astype(np.float32)

    # 期望的 dy（NCHW）
    dy_np = (2.0 / n) * (yv - tgt.numpy())
    dyb = ex._cached("dbg.dy", (n,))
    cl.enqueue_copy(ex.dev.q, dyb, np.ascontiguousarray(dy_np.reshape(-1) * SCALE, np.float16))
    ex.dev.q.finish()

    # ---- 设备侧：conv_out 这一层的 dy（NHWC）与它的 db ----
    rec = [r for r in ex.recs if r[0] == "conv2d"][-1]
    layer, (N, OH, OW) = rec[5], rec[6]
    OC = rec[3].shape[0]
    ex.backward(dyb, y.shape)
    ex.dev.q.finish()
    dy_dev = ex.dev.down(layer.dy, (N, OH, OW, OC)).astype(np.float32) / SCALE
    dy_ref = np.transpose(dy_np, (0, 2, 3, 1))
    print("  1) dy 转换: rel %.2e" % (np.linalg.norm(dy_dev - dy_ref) / np.linalg.norm(dy_ref)))

    # ---- torch 参考（只训 conv_out）----
    ref = build_encoder(r"D:\work\textmodel\_sd15_combined\vae", 64)
    ref.load_state_dict(enc.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)
    for p in ref.conv_out.parameters():
        p.requires_grad_(True)
    ((ref(x) - tgt) ** 2).mean().backward()

    # ---- conv_out 的权重/偏置梯度 ----
    pw, pb = ex.params["conv_out.weight"], ex.params["conv_out.bias"]
    gw = ex.dev.down(pw.grad, pw.shape).astype(np.float32) / SCALE
    gb = ex.dev.down(pb.grad, pb.shape).astype(np.float32) / SCALE
    tw = ref.conv_out.weight.grad.numpy().reshape(gw.shape)
    tb = ref.conv_out.bias.grad.numpy()
    print("  2) conv_out.weight grad: dev %.5f torch %.5f rel %.2e"
          % (np.linalg.norm(gw), np.linalg.norm(tw),
             np.linalg.norm(gw - tw) / (np.linalg.norm(tw) + 1e-12)))
    print("  3) conv_out.bias   grad: dev %.5f torch %.5f rel %.2e"
          % (np.linalg.norm(gb), np.linalg.norm(tb),
             np.linalg.norm(gb - tb) / (np.linalg.norm(tb) + 1e-12)))

    # ---- 底座权重是否一致（frozen 只有 fp16 副本）----
    for nm in ("conv_in.weight", "down_blocks.0.resnets.0.conv1.weight"):
        p = ex.params.get(nm)
        if p is None:
            continue
        wd = ex.dev.down(p.buf, p.shape).astype(np.float32)
        wt = enc.state_dict()[nm].numpy().reshape(p.shape)
        print("  4) %s（frozen）: rel %.2e" % (nm, np.linalg.norm(wd - wt) / np.linalg.norm(wt)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
