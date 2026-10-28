# -*- coding: utf-8 -*-
"""bench_optimizer.py — effective memory bandwidth of CPU optimizers.

The element-wise ops in training (optimizer update dominates once GEMMs are fast) are
memory-bound: on this laptop STREAM-like torch ops reached 12-20 GB/s. This measures the
bytes a real optimizer step must move and the GB/s actually achieved, for:
  * bitsandbytes AdamW8bit (1-byte states + per-256 absmax), and
  * torch AdamW (fp32 m + v, reference).

Effective bytes per element:
  AdamW8bit : read p(4) + write p(4) + read g(4) + r/w state1(1+1) + r/w state2(1+1)
              + absmax (~4/256) ~= 16.02 B/elem
  AdamW(fp32): read p(4) + write p(4) + read g(4) + r/w m(8) + r/w v(8) = 28 B/elem

Usage: python bench_optimizer.py [--quick]
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import os
import statistics
import sys
import time

import torch

_IMPORT_ROOT = os.environ.get("BNB_PKG_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bitsandbytes")
sys.path.insert(0, _IMPORT_ROOT)
import bitsandbytes as bnb  # noqa: E402

MB = 1 << 20

BPE = {
    # read p(4) + write p(4) + read g(4) + two 1-byte states read+write (4) + absmax
    "AdamW8bit": 4 + 4 + 4 + 4 + 4 / 256,
    # p r/w(8) + g read(4) + m r/w(8) + v r/w(8)
    "AdamW-fp32": 4 + 4 + 4 + 4 + 4 + 4 + 4,
}


def measure(opt_name: str, n: int, steps: int, threads: int):
    torch.set_num_threads(threads)
    p = torch.nn.Parameter(torch.randn(n))
    p.grad = torch.randn(n)
    if opt_name == "AdamW8bit":
        opt = bnb.optim.AdamW8bit([p], lr=1e-4)
    else:
        opt = torch.optim.AdamW([p], lr=1e-4)

    for _ in range(2):  # warmup
        opt.step()
        opt.zero_grad(set_to_none=False)
        p.grad = torch.randn(n)

    times = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        p.grad = torch.randn(n)
        t0 = time.perf_counter()
        opt.step()
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    gbs = n * BPE[opt_name] / med / 1e9
    return med, gbs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    sizes = [4 * 1024 * 1024, 16 * 1024 * 1024]
    if not args.quick:
        sizes.append(64 * 1024 * 1024)
    steps = 5
    cores = os.cpu_count() or 6
    print("=" * 78)
    print("optimizer bandwidth | torch %s | %d cores" % (torch.__version__, cores))
    print("=" * 78)
    print("%-10s %-12s %-6s %8s %10s %8s" %
          ("optimizer", "params(M)", "threads", "ms/step", "GB/s", "state(MB)"))
    rows = []
    for opt_name in ("AdamW8bit", "AdamW-fp32"):
        for n in sizes:
            for threads in (1, cores):
                med, gbs = measure(opt_name, n, steps, threads)
                state_bytes = (2 * n + (n // 256) * 4) if opt_name == "AdamW8bit" else (8 * n)
                rows.append((opt_name, n, threads, med, gbs, state_bytes))
                print("%-10s %-12.0f %-6d %8.2f %10.2f %8.0f" %
                      (opt_name, n / 1e6, threads, med * 1e3, gbs, state_bytes / MB))
    print()
    print("BENCH DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
