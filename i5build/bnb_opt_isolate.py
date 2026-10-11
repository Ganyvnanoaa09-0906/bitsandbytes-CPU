"""Isolate the optimizer step from forward/backward and find out why 8-bit is slow.

The first benchmark timed zero_grad + forward + backward + step together, so a 3x gap
could have come from anywhere. Here the backward is done once and then only opt.step()
is timed, with a fixed .grad in place.

Also worth ruling out before blaming the kernel: per-call overhead (the LUT cache keys
on the qmap pointer), thread scaling, and whether the AVX2 path is taken at all.
"""
import time

import torch
from bitsandbytes.optim import AdamW8bit

torch.manual_seed(0)


def bench(n, threads, reps=20):
    torch.set_num_threads(threads)
    out = {}
    for name, make in (("fp32 AdamW", lambda p: torch.optim.AdamW([p], lr=1e-4)),
                       ("8-bit AdamW", lambda p: AdamW8bit([p], lr=1e-4))):
        p = torch.nn.Parameter(torch.randn(n) * 0.05)
        p.grad = torch.randn(n) * 0.01
        opt = make(p)
        for _ in range(3):                       # warm caches and the LUT
            opt.step()
        t0 = time.perf_counter()
        for _ in range(reps):
            opt.step()
        dt = (time.perf_counter() - t0) / reps
        out[name] = dt
        del opt, p
    return out


def main():
    print(f"torch {torch.__version__}, threads sweep")
    print(f"\n{'params':>12} {'threads':>8} {'fp32 ms':>10} {'8bit ms':>10} "
          f"{'ratio':>7} {'fp32 Melem/s':>13} {'8bit Melem/s':>13}")
    for n in (1 << 20, 8 << 20, 33 << 20):
        for th in (1, 2, 6):
            r = bench(n, th)
            f, e = r["fp32 AdamW"], r["8-bit AdamW"]
            print(f"{n/1e6:11.1f}M {th:8d} {f*1e3:10.2f} {e*1e3:10.2f} "
                  f"{e/f:6.2f}x {n/f/1e6:12.1f} {n/e/1e6:12.1f}")

    print("\n=== where the 8-bit step goes (33 M params, 6 threads) ===")
    torch.set_num_threads(6)
    n = 33 << 20
    p = torch.nn.Parameter(torch.randn(n) * 0.05)
    p.grad = torch.randn(n) * 0.01
    opt = AdamW8bit([p], lr=1e-4)
    opt.step()
    # state layout tells us the block count and therefore the work per block
    st = next(iter(opt.state.values()))
    for k, v in st.items():
        if torch.is_tensor(v):
            print(f"  {k:<10} {tuple(v.shape)}  {v.dtype}")


if __name__ == "__main__":
    main()
