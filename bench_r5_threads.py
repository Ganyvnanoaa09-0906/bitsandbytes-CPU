# bench_r5_threads.py - R5-4500U (6C6T AVX2) 线程校准基准
# 测：GEMM / Conv / attention 在 4/5/6 线程下的吞吐，决定文本训练最优线程数
# 运行：py -3.11 bench_r5_threads.py
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import torch
import torch.nn as nn


def bench_gemm(threads, M=512, K=2048, N=8192, iters=6, warm=2):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    for _ in range(warm):
        c = a @ b
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    dt = (time.perf_counter() - t0) / iters
    return 2 * M * K * N / dt / 1e9


def bench_conv(threads, C_in=256, C_out=256, H=64, W=64, k=3, iters=6, warm=2):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    x = torch.randn(1, C_in, H, W)
    conv = nn.Conv2d(C_in, C_out, k, padding=k // 2).eval()
    for _ in range(warm):
        conv(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        conv(x)
    dt = (time.perf_counter() - t0) / iters
    return 2 * C_in * C_out * k * k * H * W / dt / 1e9


def bench_attn(threads, B=8, Hh=8, L=1024, D=64, iters=5, warm=1):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    q = torch.randn(B, Hh, L, D)
    k = torch.randn(B, Hh, L, D)
    v = torch.randn(B, Hh, L, D)
    for _ in range(warm):
        (q @ k.transpose(-1, -2)) @ v
    t0 = time.perf_counter()
    for _ in range(iters):
        (q @ k.transpose(-1, -2)) @ v
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    print("=== R5-4500U 线程校准 (AVX2, fp32) ===")
    print(f"capability {torch.backends.cpu.get_cpu_capability()}  "
          f"oneDNN {torch.backends.mkldnn.enabled}  default threads {torch.get_num_threads()}\n")

    print("--- GEMM 512x2048x8192 (GFLOP/s) ---")
    for t in (4, 5, 6):
        try:
            print(f"  {t} 线程: {bench_gemm(t):6.0f} GFLOP/s")
        except Exception as e:
            print(f"  {t} 线程: FAIL {e}")

    print("--- Conv 256ch 3x3 64x64 (GFLOP/s) ---")
    for t in (4, 5, 6):
        try:
            print(f"  {t} 线程: {bench_conv(t):6.0f} GFLOP/s")
        except Exception as e:
            print(f"  {t} 线程: FAIL {e}")

    print("--- attention 小块 (ms) ---")
    for t in (4, 5, 6):
        try:
            print(f"  {t} 线程: {bench_attn(t):6.1f} ms")
        except Exception as e:
            print(f"  {t} 线程: FAIL {e}")

    # 小 GEMM 密集场景（LoRA 训练里多是 256-1024 维小矩阵）
    print("--- 小 GEMM 512x512x512 x50 (GFLOP/s) ---")
    for t in (4, 5, 6):
        try:
            torch.set_num_threads(t)
            torch.manual_seed(0)
            a = torch.randn(512, 512)
            b = torch.randn(512, 512)
            for _ in range(2):
                a @ b
            t0 = time.perf_counter()
            for _ in range(50):
                a @ b
            dt = (time.perf_counter() - t0) / 50
            print(f"  {t} 线程: {2*512**3/dt/1e9:6.0f} GFLOP/s")
        except Exception as e:
            print(f"  {t} 线程: FAIL {e}")

    print("\ndone")


if __name__ == "__main__":
    main()
