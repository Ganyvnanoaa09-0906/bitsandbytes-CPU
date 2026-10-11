"""How many threads should the i5 actually use? Measure the op that dominates.

The i5-10400 has 6 physical cores and 12 logical ones, and the render was run with 12
because `detect_cpu.py` recommends `logical` for diffusion. That recommendation is a
heuristic, not a measurement, and SMT shares the FP units: for a compute-bound fp32 GEMM
the second thread on a core often adds nothing, and can subtract by halving the L1/L2
each thread gets.

The profiler already said 82% of a step is `aten::mm`, so this measures exactly that, at
the shapes the model uses, across thread counts. A 20-minute answer to a question that
would otherwise cost two 75-minute renders.

Usage:
    python torch_thread_scale.py
"""
import time

import torch

torch.set_grad_enabled(False)

SHAPES = [(16384, 4096, 4096), (16384, 4096, 12288), (4096, 4096, 4096)]


def bench(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    import os
    print(f"torch {torch.__version__}  logical cpus {os.cpu_count()}  "
          f"OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS', '<unset>')}")
    try:
        import psutil
        print(f"physical cores {psutil.cpu_count(logical=False)}")
    except Exception:
        pass

    mats = {}
    for M, K, N in SHAPES:
        mats[(M, K, N)] = (torch.randn(M, K), torch.randn(K, N))

    counts = [3, 6, 8, 12]
    print(f"\n{'shape':>22} " + " ".join(f"{c:>9}T" for c in counts) + "   best")
    results = {}
    for shape in SHAPES:
        M, K, N = shape
        a, b = mats[shape]
        row = []
        for c in counts:
            torch.set_num_threads(c)
            dt = bench(lambda: a @ b)
            row.append(dt)
            results[(shape, c)] = dt
        best = min(range(len(counts)), key=lambda i: row[i])
        print(f"{f'{M}x{K}x{N}':>22} " + " ".join(f"{t:10.3f}" for t in row) +
              f"   {counts[best]} threads ({row[best]:.3f} s)")
        del a, b

    print("\n=== GFLOPS ===")
    print(f"{'shape':>22} " + " ".join(f"{c:>9}T" for c in counts))
    for shape in SHAPES:
        M, K, N = shape
        flops = 2.0 * M * K * N
        print(f"{f'{M}x{K}x{N}':>22} " +
              " ".join(f"{flops/results[(shape, c)]/1e9:10.1f}" for c in counts))

    print("\n=== the dequantise kernel, which uses its own OpenMP (not torch's) ===")
    from bitsandbytes.nn import Linear4bit, Params4bit
    import torch.nn as nn
    lin = nn.Linear(4096, 4096, bias=False)
    q = Linear4bit(4096, 4096, bias=False, compute_dtype=torch.bfloat16,
                   quant_type="nf4", compress_statistics=True)
    q.weight = Params4bit(lin.weight.data, requires_grad=False, quant_type="nf4",
                          compress_statistics=True)
    q.weight = q.weight.to("cpu")
    x1 = torch.randn(1, 4096, dtype=torch.bfloat16)      # decode shape
    xN = torch.randn(16384, 4096, dtype=torch.bfloat16)  # diffusion shape
    for name, x in (("M=1 (decode)", x1), ("M=16384 (diffusion)", xN)):
        row = []
        for c in counts:
            torch.set_num_threads(c)
            row.append(bench(lambda: q(x)))
        print(f"  {name:<22} " + " ".join(f"{c}T={t*1e3:8.1f}ms" for c, t in zip(counts, row)))
    print("\n  Linear4bit uses the library's own OpenMP region, so torch.set_num_threads")
    print("  may not govern it at all -- which is why the same sweep is worth running")
    print("  with OMP_NUM_THREADS set differently in the environment.")


if __name__ == "__main__":
    main()
