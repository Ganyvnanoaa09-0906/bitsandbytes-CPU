# bench_cpu_avx2.py - 小规模合成基准：i5-10400 (AVX2) 上 PyTorch CPU 计算吞吐
# 目的：给出"训练速度最快"的配置结论（线程数 × dtype），不加载大模型、不烧内存。
#
# 运行：py -3.11 bench_cpu_avx2.py
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import torch

# ---------- 合成 FFN 块（模拟 1.7B 模型每层的密集 GEMM：hidden 2048, ff 8192） ----------
def bench_gemm(threads, dtype, hidden=2048, ff=8192, iters=6, warmup=2):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    a = torch.randn(512, hidden, dtype=dtype)
    w = torch.randn(hidden, ff, dtype=dtype)
    # warmup
    for _ in range(warmup):
        b = a @ w
    t0 = time.time()
    for _ in range(iters):
        b = a @ w
        _ = b.sum().item()
    dt = (time.time() - t0) / iters
    gflops = 512 * hidden * ff * 2 / dt / 1e9
    return dt, gflops


def bench_fwd_bwd(threads, dtype, hidden=2048, ff=8192, layers=4, seq=512, iters=3, warmup=1):
    """模拟一层 transformer 的 fwd+bwd 计算密度（两层 linear + 激活 + layernorm）。"""
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    x = torch.randn(seq, hidden, dtype=dtype)
    w1 = torch.randn(hidden, ff, dtype=dtype)
    w2 = torch.randn(ff, hidden, dtype=dtype)
    b1 = torch.randn(ff, dtype=dtype)
    b2 = torch.randn(hidden, dtype=dtype)
    for _ in range(warmup):
        y = torch.nn.functional.gelu(x @ w1 + b1) @ w2 + b2
        y.mean().backward()
    t0 = time.time()
    for _ in range(iters):
        x.grad = None; w1.grad = None; w2.grad = None
        y = torch.nn.functional.gelu(x @ w1 + b1) @ w2 + b2
        (y.mean() * 0.0 + y.sum() * 0.5).backward()
    dt = (time.time() - t0) / iters
    return dt


print("=== i5-10400 (AVX2) PyTorch CPU 合成基准 ===")
print(f"capability: {torch.backends.cpu.get_cpu_capability()}, logical cores {os.cpu_count()}")
print(f"oneDNN(mkldnn): {torch.backends.mkldnn.enabled}, f32 matmul: {torch.get_float32_matmul_precision()}\n")

for threads in (6, 8, 12):
    for dtype in (torch.float32, torch.bfloat16):
        dt, g = bench_gemm(threads, dtype)
        print(f"GEMM 512x2048x8192  threads={threads:2d}  {str(dtype).split('.')[-1]:8s}  "
              f"{dt*1e3:7.1f} ms   {g:6.0f} GFLOP/s")
print()

for threads in (6, 12):
    dt32 = bench_fwd_bwd(threads, torch.float32)
    dtbf = bench_fwd_bwd(threads, torch.bfloat16)
    print(f"合成 FFN fwd+bwd (seq=512, 4块)  threads={threads:2d}  fp32 {dt32*1e3:7.1f} ms   "
          f"bf16 {dtbf*1e3:7.1f} ms   (bf16/fp32 {dtbf/dt32:.2f}x)")
