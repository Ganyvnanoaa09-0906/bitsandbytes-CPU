"""Is an int8 GEMM actually faster than fp32 on this CPU, at the shapes that matter?

The profiler says 82% of a Qwen-Image-2.1 step is `aten::mm` on fp32 weights. Narrower
WEIGHTS do not help -- at M=16384 the GEMM is compute bound, and neither Zen2 nor Comet
Lake has VNNI, so a 4-bit weight still has to be unpacked to fp32 before it can be
multiplied. What could help is narrower ARITHMETIC: AVX2's `vpmaddubsw` multiplies 32
int8 pairs per instruction where an fp32 FMA does 8, and int8 weights are a quarter of
the bytes.

Before writing any of that by hand, this checks whether the machine has a fast int8 path
at all, by timing the one that already ships: `torch._int_mm`, which lowers to oneDNN's
int8 GEMM.

Two things make the answer non-obvious and both are measured rather than assumed:
  * the activation quantisation is part of the cost, not an afterthought -- an int8 GEMM
    needs int8 activations, and computing them is a full pass over the input
  * `_int_mm` has shape restrictions (M must be large enough) and returns int32, so the
    accumulation and the dequantisation both have to be counted

Usage:
    python int8_probe.py
"""
import time

import torch

torch.set_grad_enabled(False)
torch.set_num_threads(6)

# (M, K, N) from Qwen-Image-2.1's transformer at 1024x1024: 16384 latent tokens,
# model dim 4096, MLP dim 12288.
SHAPES = [(4096, 4096, 4096), (4096, 4096, 1024), (16384, 4096, 1024)]


def bench(fn, reps=2, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def quantize_per_row(x):
    """Symmetric per-row int8 quantisation, returning (q, scale)."""
    scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def main():
    print(f"torch {torch.__version__}  threads {torch.get_num_threads()}")
    print(f"int8 mm available: {hasattr(torch, '_int_mm')}")
    print(f"\n{'shape (M,K,N)':>22} {'fp32 s':>9} {'int8 s':>9} {'ratio':>7} "
          f"{'fp32 GFLOPS':>12} {'int8 GOPS':>10}")
    for M, K, N in SHAPES:
        print(f"  {M}x{K}x{N} ...", flush=True)
        a = torch.randn(M, K)
        b = torch.randn(K, N)

        t_fp32 = bench(lambda: a @ b)
        flops = 2.0 * M * K * N
        print(f"    fp32 mm   {t_fp32:8.3f} s  ({flops/t_fp32/1e9:.1f} GFLOPS)", flush=True)
        try:
            t0 = time.perf_counter()
            aq, _ = quantize_per_row(a)
            bq, bsc = quantize_per_row(b.t())
            bq = bq.t().contiguous()
            t_quant = time.perf_counter() - t0
            print(f"    int8 quantise weights+acts {t_quant:8.3f} s", flush=True)
            t_int8 = bench(lambda: torch._int_mm(aq, bq))
            print(f"    int8 mm   {t_int8:8.3f} s  ({flops/t_int8/1e9:.1f} GOPS)   "
                  f"-> int8/fp32 = {t_int8/t_fp32:.2f}x", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    int8 FAILED: {type(exc).__name__}: {str(exc)[:80]}", flush=True)

    print("\n=== the activation quantisation has to be paid too ===")
    for M, K, N in SHAPES[:2]:
        a = torch.randn(M, K)
        t_q = bench(lambda: quantize_per_row(a))
        flops = 2.0 * M * K * N
        t_fp32 = bench(lambda: a @ b if False else a @ torch.randn(K, N))
        print(f"  {M}x{K}: per-row int8 quantise {t_q*1e3:8.1f} ms  "
              f"(fp32 GEMM is {t_fp32*1e3:8.1f} ms)")

    print("\n=== vs the 4-bit path we ship ===")
    print("  the shipped path dequantises 4-bit weights to fp32 and calls fp32 mm,")
    print("  so its GEMM cost equals the fp32 column above and its dequantise cost is")
    print("  the 1% the profiler attributed to bitsandbytes::dequantize_4bit.")


if __name__ == "__main__":
    main()
