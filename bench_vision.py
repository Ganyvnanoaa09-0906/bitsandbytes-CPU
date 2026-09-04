# bench_vision.py - 生图/视觉模型训练的 CPU 基准（i5-10400 AVX2）
# 测：① 2D 卷积吞吐（oneDNN AVX2）② 8bit 量化 conv 权重可行性（内存/开销）
# 运行：py -3.11 bench_vision.py
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn

# 本地 bnb（相对本脚本定位，双机迁移免改）
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

print("=== 生图/视觉 CPU 基准 (AVX2) ===")
print(f"capability {torch.backends.cpu.get_cpu_capability()}  oneDNN {torch.backends.mkldnn.enabled}\n")


def bench_conv(threads, C_in=256, C_out=256, H=64, W=64, k=3, iters=6, warm=2):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    x = torch.randn(1, C_in, H, W)
    conv = nn.Conv2d(C_in, C_out, k, padding=k // 2).eval()
    for _ in range(warm):
        conv(x)
    t0 = time.time()
    for _ in range(iters):
        conv(x)
    dt = (time.time() - t0) / iters
    flops = 2 * C_in * C_out * k * k * H * W  # per-image
    return dt * 1e3, flops / dt / 1e9


def bench_attn(threads, B=8, Hh=8, L=1024, D=64, iters=5, warm=1):
    """latent diffusion 的 self-attn（qk^T 是 GEMM）。"""
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    q = torch.randn(B, Hh, L, D); k = torch.randn(B, Hh, L, D); v = torch.randn(B, Hh, L, D)
    for _ in range(warm):
        (q @ k.transpose(-1, -2)) @ v
    t0 = time.time()
    for _ in range(iters):
        (q @ k.transpose(-1, -2)) @ v
    dt = (time.time() - t0) / iters
    return dt * 1e3


print("--- 2D 卷积 (U-Net 50% 尺寸: 256ch, 3x3, 64x64) ---")
for t in (6, 8, 12):
    dt, g = bench_conv(t)
    print(f"  threads={t:2d}  {dt:7.1f} ms  {g:6.0f} GFLOP/s")

print("\n--- latent self-attn (8x8x1024x64) ---")
for t in (6, 8, 12):
    print(f"  threads={t:2d}  {bench_attn(t):7.1f} ms")

print("\n--- 8bit 量化 conv 权重 (内存/开销) ---")
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise  # noqa: E402

w = torch.randn(C_in := 256, C_out := 256, 3, 3)
w2 = w.reshape(-1)
wq, absmax = quantize_blockwise(w2, blocksize=256)
wd = dequantize_blockwise(wq, absmax, blocksize=256)
mem_fp = w.numel() * 4
mem_8 = wq.numel() * 1 + absmax.numel() * 4
print(f"  conv 权重 {C_in}ch: fp32 {mem_fp/1e6:.1f} MB -> 8bit {mem_8/1e6:.1f} MB (省 {100*(1-mem_8/mem_fp):.0f}%)")
print(f"  dequantize 误差 {abs((wd-w2).abs().max()):.2e}")

# 8bit 前向开销：dequantize + conv
conv_fp = nn.Conv2d(256, 256, 3, padding=1)
conv_fp.weight.data = w; conv_fp.eval()
def conv_8bit_fwd(x):
    wd = dequantize_blockwise(wq, absmax, blocksize=256).reshape(256, 256, 3, 3)
    return torch.nn.functional.conv2d(x, wd, padding=1)
x = torch.randn(1, 256, 64, 64)
t0 = time.time()
for _ in range(5): conv_fp(x)
dt_fp = (time.time() - t0) / 5
t0 = time.time()
for _ in range(5): conv_8bit_fwd(x)
dt_8 = (time.time() - t0) / 5
print(f"  conv 前向 fp32 {dt_fp*1e3:.1f} ms | 8bit(dequant+conv) {dt_8*1e3:.1f} ms (慢 {dt_8/dt_fp:.2f}x)")
