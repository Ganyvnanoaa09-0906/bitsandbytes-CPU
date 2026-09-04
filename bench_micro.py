# bench_micro.py - 魔改库 C 内核微基准（GDN / 量化 / 8bit 优化器）
# 用法：py -3.11 bench_micro.py [repeat]
# 输出每项耗时（秒），用于对比不同编译参数（/favor:AMD64 vs INTEL64）的 DLL
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
sys.stdout.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import torch  # noqa: E402

torch.set_num_threads(6)
torch.manual_seed(0)

import bitsandbytes as bnb  # noqa: E402
from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule  # noqa: E402
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise  # noqa: E402

REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5

print(f"=== 微基准 (repeat={REPEAT}) ===")
print(f"dll: {bnb.gdn_cpu.load_native()._name}")

# 1) GDN fwd+bwd（模拟 Qwen3-Next 一层：B=2 T=1024 H=4 K=V=64）
B, T, H, K, V = 2, 1024, 4, 64, 64
q = torch.randn(B, T, H, K, dtype=torch.float32) * K ** -0.5
k = torch.randn(B, T, H, K, dtype=torch.float32) * K ** -0.5
v = torch.randn(B, T, H, V, dtype=torch.float32) * 0.3
beta = torch.rand(B, T, H, dtype=torch.float32) * 0.8 + 0.2
g = -torch.rand(B, T, H, dtype=torch.float32) * 1.5 - 0.05

t0 = time.perf_counter()
for _ in range(REPEAT):
    qa, ka, va = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
    o, _ = fused_recurrent_gated_delta_rule(qa, ka, va, beta, g, output_final_state=False)
    o.sum().backward()
dt = (time.perf_counter() - t0) / REPEAT
print(f"GDN fwd+bwd (T=1024): {dt*1e3:.1f} ms/次")

# 2) 量化 + 反量化（4096x4096 fp32 权重）
w = torch.randn(4096, 4096)
t0 = time.perf_counter()
for _ in range(REPEAT):
    wq, absmax = quantize_blockwise(w, blocksize=256)
    wd = dequantize_blockwise(wq, absmax, blocksize=256)
dt = (time.perf_counter() - t0) / REPEAT
print(f"quant+dequant 4096x4096: {dt*1e3:.1f} ms/次")

# 3) 8bit AdamW 更新（4M 参数，1 步）
n = 4_000_000
p = torch.randn(n, requires_grad=True)
g_ = torch.randn(n)
opt = bnb.optim.AdamW8bit([p], lr=1e-4)
t0 = time.perf_counter()
for _ in range(REPEAT):
    p.grad = g_
    opt.step()
dt = (time.perf_counter() - t0) / REPEAT
print(f"AdamW8bit 更新 4M 参数: {dt*1e3:.1f} ms/步")

# 4) 8bit AdamW 小参数（LoRA 规模 3.2M）
n2 = 3_200_000
p2 = torch.randn(n2, requires_grad=True)
g2 = torch.randn(n2)
opt2 = bnb.optim.AdamW8bit([p2], lr=1e-4)
t0 = time.perf_counter()
for _ in range(REPEAT):
    p2.grad = g2
    opt2.step()
dt = (time.perf_counter() - t0) / REPEAT
print(f"AdamW8bit 更新 3.2M 参数: {dt*1e3:.1f} ms/步")

print("done")
