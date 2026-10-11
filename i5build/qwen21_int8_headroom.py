"""Is there a wider arithmetic path than fp32 on this CPU, on THIS model's shapes?

The Qwen-Image-2.1 step is 84% aten::mm and those matmuls run at ~84% of the machine's
measured fp32 GEMM ceiling, so no amount of scheduling, thread tuning or fusion of the
dequantise step can move the number much. The only remaining lever is doing the same
multiply in a wider integer domain: AVX2 has no VNNI, but it does have vpmaddubsw, and
this project already measured 40.4 int8 MAC/cycle against 10.6 fp32 MAC/cycle at the
register level.

What matters here is not the microbenchmark but the library: if PyTorch/oneDNN already
has a fast int8 GEMM on AVX2, integrating it is a one-line change; if it does not, the
only route is the hand-written kernel in kernels/int8_gemm.c.

The shapes are the ones the model actually calls, read off the probe:
    img_mlp    (M, 12288, 4096) and (M, 4096, 12288)
    attention  (M, 4096, 4096)
with M = 1054 at 512x512 and M = 4126 at 1024x1024 (image tokens + text tokens).
"""
import time

import torch

torch.set_num_threads(6)


def bench(fn, reps=5, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def fp32_mm(m, n, k):
    x = torch.randn(m, k)
    w = torch.randn(n, k)
    dt = bench(lambda: torch.mm(x, w.t()))
    return dt, 2.0 * m * n * k / dt / 1e9


def int8_mm(m, n, k):
    """torch._int_mm: int8 x int8 -> int32. OneDNN's path, if this CPU has one."""
    a = torch.randint(-100, 100, (m, k), dtype=torch.int8)
    b = torch.randint(-100, 100, (k, n), dtype=torch.int8)
    dt = bench(lambda: torch._int_mm(a, b))
    return dt, 2.0 * m * n * k / dt / 1e9


def quantized_dynamic(m, n, k):
    """The eager int8 path: dynamic quantization of activations + fbgemm/qnnpack."""
    lin = torch.nn.quantized.dynamic.Linear(k, n, dtype=torch.qint8)
    x = torch.randn(m, k)
    dt = bench(lambda: lin(x))
    return dt, 2.0 * m * n * k / dt / 1e9


print(f"torch {torch.__version__}  threads {torch.get_num_threads()}")
print(f"{'shape':<28} {'fp32 ms':>9} {'GFLOPS':>8}   {'int8 ms':>9} {'GFLOPS':>8}  ratio")
for m in (1054, 4126):
    for (n, k) in ((12288, 4096), (4096, 12288), (4096, 4096)):
        t32, g32 = fp32_mm(m, n, k)
        try:
            t8, g8 = int8_mm(m, n, k)
            s8 = f"{t8*1000:>9.1f} {g8:>8.1f}  {g8/g32:>5.2f}x"
        except Exception as exc:  # noqa: BLE001
            s8 = f"  unavailable: {type(exc).__name__}: {str(exc)[:40]}"
        print(f"  {m}x{n}x{k:<16} {t32*1000:>9.1f} {g32:>8.1f}   {s8}")

print()
print("dynamic quantization (fbgemm/qnnpack eager path), same shapes:")
for m in (1054,):
    for (n, k) in ((12288, 4096), (4096, 4096)):
        t32, g32 = fp32_mm(m, n, k)
        try:
            tq, gq = quantized_dynamic(m, n, k)
            print(f"  {m}x{n}x{k:<16} fp32 {g32:>6.1f} GFLOPS   int8 {gq:>6.1f} GFLOPS"
                  f"  {gq/g32:>5.2f}x")
        except Exception as exc:  # noqa: BLE001
            print(f"  {m}x{n}x{k:<16} unavailable: {type(exc).__name__}: "
                  f"{str(exc)[:60]}")
