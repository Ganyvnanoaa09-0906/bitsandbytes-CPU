# -*- coding: utf-8 -*-
"""check_opencl_suite.py — 驱动更新后的 OpenCL 功能巡检（驱动更新/换机后跑这个）（不涉及 UNet 整链，逐原语对拍）。"""
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\gt1030")
import numpy as np
import pyopencl as cl
import torch

import gt1030.igpu_ctx as C
from gt1030.igpu_cat import cat2
from gt1030.igpu_conv import Conv2d, K as CVK
from gt1030.igpu_gn import GroupNorm

dev = C.DevCtx(arena_mb=1200, name="suite").activate()
rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
rows = []

# 1) GroupNorm
N, Ch, H, W = 1, 320, 16, 16
x = torch.randn(N, Ch, H, W)
gn = torch.nn.GroupNorm(32, Ch, eps=1e-5).eval()
with torch.no_grad():
    ref = gn(x).numpy()
g = GroupNorm(dev, N, Ch, H, W, 32, eps=1e-5)
cl.enqueue_copy(dev.q, g.gamma, np.ascontiguousarray(gn.weight.detach().numpy().astype(np.float16)))
cl.enqueue_copy(dev.q, g.beta, np.ascontiguousarray(gn.bias.detach().numpy().astype(np.float16)))
xb = dev.up(x.numpy().astype(np.float16))
y = dev.buf(N * Ch * H * W * 2)
g.forward(xb, y)
dev.q.finish()
rows.append(("GroupNorm", rel(dev.down(y, (N, Ch, H, W)).astype(np.float32), ref)))

# 2) SiLU
from gt1030.igpu_unet_t2d_dev import K as T2DK

s_src = dev.up((np.random.randn(1 << 16) * 2).astype(np.float16))
s_dst = dev.buf((1 << 16) * 2)
T2DK(dev)["silu_fwd"](dev.q, (1 << 16,), None, s_src, s_dst, np.int64(1 << 16))
dev.q.finish()
got = dev.down(s_dst, (1 << 16,)).astype(np.float32)
src = dev.down(s_src, (1 << 16,)).astype(np.float32)
rows.append(("SiLU", rel(got, src / (1 + np.exp(-src)))))

# 3) cat2
a = torch.randn(1, 320, 16, 16)
b2 = torch.randn(1, 320, 16, 16)
ab = dev.up(a.numpy().astype(np.float16))
bb = dev.up(b2.numpy().astype(np.float16))
dst = dev.buf(1 * 640 * 16 * 16 * 2)
cat2(dev, dst, ab, bb, 1, 320, 320, 16, 16)
dev.q.finish()
ref_cat = torch.cat([a, b2], dim=1).numpy()
rows.append(("cat2", rel(dev.down(dst, (1, 640, 16, 16)).astype(np.float32), ref_cat)))

# 4) upsample2x（nearest，NCHW）
up_in = torch.randn(1, 64, 8, 8)
upb = dev.up(up_in.numpy().astype(np.float16))
upo = dev.buf(1 * 64 * 16 * 16 * 2)
CVK(dev)["upsample2x"](dev.q, (1 * 64 * 16 * 16,), None, upb, upo,
                       np.int32(1), np.int32(64), np.int32(8), np.int32(8))
dev.q.finish()
ref_up = torch.nn.functional.interpolate(up_in, scale_factor=2, mode="nearest").numpy()
rows.append(("upsample2x", rel(dev.down(upo, (1, 64, 16, 16)).astype(np.float32), ref_up)))

# 5) 时间嵌入（sinusoidal + 2 层 Linear）
from diffusers import UNet2DConditionModel
from gt1030.igpu_time_emb import sinusoidal
from gt1030.igpu_unet_t2d_dev import token_linear
import gt1030.igpu_unet_full_dev as F

u = UNet2DConditionModel.from_pretrained(r"D:\work\textmodel\_sd15_combined\unet").eval()
D = u.config.block_out_channels[-1]
tdim = u.config.block_out_channels[0]
with torch.no_grad():
    ref_te = u.time_embedding(u.time_proj(torch.tensor([500.0]))).numpy()
wc = F.W(dev)
t32 = np.array([500.0], np.float32)
tb = dev.buf(t32.nbytes)
cl.enqueue_copy(dev.q, tb, t32)
sb = F._buf(dev, tdim)
sinusoidal(dev, tb, sb, 1, tdim)
tw1, tb1 = wc.lin(u.time_embedding.linear_1)
n1 = u.time_embedding.linear_1.out_features
o1 = F._buf(dev, n1)
token_linear(dev, sb, tw1, tb1, o1, 1, n1, tdim)
a1 = F._buf(dev, n1)
T2DK(dev)["silu_fwd"](dev.q, (n1,), None, o1, a1, np.int64(n1))
tw2, tb2 = wc.lin(u.time_embedding.linear_2)
o2 = F._buf(dev, D)
token_linear(dev, a1, tw2, tb2, o2, 1, D, u.time_embedding.linear_2.in_features)
dev.q.finish()
rows.append(("time_embedding", rel(dev.down(o2, (1, D)).astype(np.float32), ref_te)))

print("  %-16s %-12s %s" % ("原语", "rel L2", "判定"))
for name, r in rows:
    print("  %-16s %-12.2e %s" % (name, r, "OK" if r < 3e-2 else "**FAIL**"))
bad = [n for n, r in rows if r >= 3e-2]
print("  RESULT " + ("PASS" if not bad else "FAIL: " + ", ".join(bad)))
