"""CPU-only Muon NS5 throughput for the AR-30.5M matrix families at 1/2/4/6 threads."""
from __future__ import annotations

import json
import statistics
import sys
import time

import torch

A_, B_, C_ = 3.4445, -4.7750, 2.0315
NS_STEPS = 5
FAMILIES = [((512, 512), 20), ((256, 512), 20), ((2736, 512), 10), ((512, 1368), 10)]


def ns5_torch(x):
    for _ in range(NS_STEPS):
        a = x @ x.T
        bm = B_ * a + C_ * (a @ a)
        x = A_ * x + bm @ x
    return x


def flops(o, i):
    m, n = min(o, i), max(o, i)
    return NS_STEPS * (4.0 * m * m * n + 2.0 * m * m * m)


def main():
    mats = []
    total_flops = 0.0
    for (o, i), c in FAMILIES:
        m, n = min(o, i), max(o, i)
        total_flops += flops(o, i) * c
        for _ in range(c):
            x = torch.randn(m, n)
            x = x / (x.norm() + 1e-7)
            mats.append(x)
    print('matrices=%d total_flops=%.1f GFLOP' % (len(mats), total_flops / 1e9), flush=True)
    rows = []
    for th in (1, 2, 4, 6):
        torch.set_num_threads(th)
        # warm
        ns5_torch(mats[0])
        t = time.perf_counter()
        for x in mats:
            ns5_torch(x)
        sec = time.perf_counter() - t
        row = dict(threads=th, seconds=sec, gflops=total_flops / 1e9 / sec,
                   projected_muon_step_s=6.42 - 2.42 + sec)
        rows.append(row)
        print('threads=%d  %.3f s  %.1f GFLOPS  projected_muon_step=%.2f s'
              % (th, sec, row['gflops'], row['projected_muon_step_s']), flush=True)
    print('RESULT ' + json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
