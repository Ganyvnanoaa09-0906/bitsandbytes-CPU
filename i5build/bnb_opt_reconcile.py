"""Reconcile two contradictory measurements of the same optimizer.

Isolated, opt.step() alone on 8.4 M params at 6 threads:
    fp32 AdamW  42.2 ms      8-bit AdamW  23.5 ms     -> 8-bit 1.8x FASTER
Inside a training step (zero_grad + forward + backward + step), same size:
    fp32 AdamW 182.2 ms      8-bit AdamW 552.6 ms    -> 8-bit 3.0x SLOWER

Both cannot describe the optimizer. The difference has to be in the parts that are not
the optimizer, and the candidate is the one the kernel comments point at: the 8-bit
step writes the updated parameter with non-temporal stores, which deliberately leave it
OUT of cache. If the next forward pass then has to pull the parameter back from DRAM,
the cost lands on the forward, not on the step -- and a benchmark that only times the
step would never see it.

This times forward+backward and step separately, in the same process, for both.
"""
import time

import torch
from bitsandbytes.optim import AdamW8bit

torch.manual_seed(0)
torch.set_num_threads(6)

N = 8 << 20
ROWS = N // 128


def run(name, make):
    p = torch.nn.Parameter(torch.randn(ROWS, 128) * 0.05)
    x = torch.randn(128, 128)
    y = torch.randn(128, ROWS)
    opt = make(p)

    def fb():
        opt.zero_grad()
        ((x @ p.t() - y) ** 2).mean().backward()

    for _ in range(3):
        fb()
        opt.step()

    reps = 15
    t_fb = t_st = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        fb()
        t1 = time.perf_counter()
        opt.step()
        t2 = time.perf_counter()
        t_fb += t1 - t0
        t_st += t2 - t1
    t_fb /= reps
    t_st /= reps
    print(f"  {name:<14} fwd+bwd {t_fb*1e3:8.2f} ms   step {t_st*1e3:8.2f} ms   "
          f"total {(t_fb+t_st)*1e3:8.2f} ms")
    del opt, p
    return t_fb, t_st


print(f"params {N/1e6:.1f} M, 6 threads, zero_grad included in fwd+bwd")
f_fb, f_st = run("fp32 AdamW", lambda p: torch.optim.AdamW([p], lr=1e-4))
e_fb, e_st = run("8-bit AdamW", lambda p: AdamW8bit([p], lr=1e-4))

print(f"\n  forward+backward: 8-bit is {e_fb/f_fb:.2f}x the fp32 time")
print(f"  step            : 8-bit is {e_st/f_st:.2f}x the fp32 time")
print(f"  whole iteration : 8-bit is {(e_fb+e_st)/(f_fb+f_st):.2f}x the fp32 time")

# The order above runs fp32 first, so allocator state differs for the second run.
# Repeat with the order reversed to separate an ordering artefact from a real effect.
print("\nreversed order:")
e_fb2, e_st2 = run("8-bit AdamW", lambda p: AdamW8bit([p], lr=1e-4))
f_fb2, f_st2 = run("fp32 AdamW", lambda p: torch.optim.AdamW([p], lr=1e-4))
print(f"  forward+backward: 8-bit is {e_fb2/f_fb2:.2f}x the fp32 time")
print(f"  step            : 8-bit is {e_st2/f_st2:.2f}x the fp32 time")
