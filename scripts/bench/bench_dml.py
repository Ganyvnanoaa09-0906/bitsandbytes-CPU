# -*- coding: utf-8 -*-
"""bench_dml.py — 核显 Vega 6 (DirectML) 实际性能测量。

理论峰值：
  6 CU × 64 FP32 通道 × 2(FMA) × 1.5 GHz ≈ 1.15 TFLOPS
  FP16 打包 ≈ 2.3 TFLOPS
而 CPU 实测（持续降频后）只有 107 GFLOPS。

但我们这个负载是「小算子多、访存密集」，DirectML 的启动开销和
算子覆盖率可能把优势吃掉。实测为准。

必须在 .venv_dml 里跑。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import time
import torch

try:
    import torch_directml as dml
    HAS_DML = True
except Exception as e:
    print('torch_directml 不可用:', e)
    HAS_DML = False

CPU = torch.device('cpu')
DEV = dml.device() if HAS_DML else CPU
print('=' * 70, flush=True)
print('torch', torch.__version__, '  DirectML:', HAS_DML, flush=True)
if HAS_DML:
    print('设备:', dml.device_name(0), flush=True)
print('=' * 70, flush=True)


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    if HAS_DML:
        try:
            torch.cuda.synchronize() if False else None
        except Exception:
            pass
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def run_matmul(dev, tag, dtype, sizes, iters=20):
    print()
    print('--- matmul %s (%s) ---' % (tag, str(dtype).replace('torch.', '')), flush=True)
    for n in sizes:
        try:
            a = torch.randn(n, n, device=dev, dtype=dtype)
            b = torch.randn(n, n, device=dev, dtype=dtype)
            if HAS_DML:
                a = a.contiguous(); b = b.contiguous()
            dt = bench(lambda: a @ b, iters=iters)
            gf = 2 * n ** 3 / dt / 1e9
            print('  n=%-6d %10.2f GFLOP/s   (%.3f ms)' % (n, gf, dt * 1e3), flush=True)
            del a, b
        except Exception as e:
            print('  n=%-6d 失败: %s' % (n, str(e)[:60]), flush=True)


# ---------- CPU 基线 ----------
torch.set_num_threads(6)
print()
print('########## CPU (6 线程, fp32) ##########', flush=True)
run_matmul(CPU, 'CPU', torch.float32, (256, 512, 1024, 2048))

# ---------- 核显 ----------
if HAS_DML:
    print()
    print('########## 核显 (DirectML) ##########', flush=True)
    run_matmul(DEV, 'DML-fp32', torch.float32, (256, 512, 1024, 2048))
    print()
    run_matmul(DEV, 'DML-fp16', torch.float16, (256, 512, 1024, 2048))

    # ---------- 真实训练形状 ----------
    print()
    print('--- 真实训练形状的 Linear (b*T=1024, 512->2048->512, 含反向) ---', flush=True)
    for tag, dev in (('CPU', CPU), ('DML', DEV)):
        try:
            torch.manual_seed(0)
            w1 = torch.randn(2048, 512, device=dev)
            w2 = torch.randn(512, 2048, device=dev)
            x = torch.randn(1024, 512, device=dev, requires_grad=True)

            def step():
                h = torch.relu(x @ w1.t())
                y = h @ w2.t()
                loss = y.pow(2).mean()
                loss.backward()

            dt = bench(step, iters=20, warm=5)
            fl = 2 * 1024 * 512 * 2048 * 2 + 2 * 1024 * 2048 * 512 * 2
            print('  %-4s  %8.3f ms   %8.1f GFLOP/s' % (tag, dt * 1e3, fl / dt / 1e9), flush=True)
            del w1, w2, x
        except Exception as e:
            print('  %-4s 失败: %s' % (tag, str(e)[:70]), flush=True)

    # ---------- 我们的模型规模（30M）能否跑 ----------
    print()
    print('--- 30M 参数模型的一步 (仿 AR block, b=8 T=256) ---', flush=True)
    import torch.nn as nn
    import torch.nn.functional as F

    class Blk(nn.Module):
        def __init__(self, d=512, f=2048):
            super().__init__()
            self.n = nn.LayerNorm(d)
            self.w1 = nn.Linear(d, f, bias=False)
            self.w3 = nn.Linear(d, f, bias=False)
            self.w2 = nn.Linear(f, d, bias=False)

        def forward(self, x):
            h = self.n(x)
            return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

    for tag, dev in (('CPU', CPU), ('DML', DEV)):
        try:
            torch.manual_seed(0)
            m = nn.Sequential(*[Blk() for _ in range(4)]).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            x = torch.randn(8, 256, 512, device=dev)

            def step():
                y = m(x)
                loss = y.pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            dt = bench(step, iters=8, warm=2)
            print('  %-4s  4 层 FFN  训练 %8.3f ms/步' % (tag, dt * 1e3), flush=True)
            del m, opt, x
        except Exception as e:
            print('  %-4s 失败: %s' % (tag, type(e).__name__ + ': ' + str(e)[:70]), flush=True)

print()
print('DONE', flush=True)
