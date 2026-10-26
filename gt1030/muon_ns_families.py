"""CPU NS5 per family and NS3 vs NS5 (for split / iteration count decisions)."""
from __future__ import annotations

import json
import time

import torch

A_, B_, C_ = 3.4445, -4.7750, 2.0315
FAMILIES = [((512, 512), 20), ((256, 512), 20), ((2736, 512), 10), ((512, 1368), 10)]
torch.set_num_threads(6)


def ns5(x, steps=5):
    for _ in range(steps):
        a = x @ x.T
        bm = B_ * a + C_ * (a @ a)
        x = A_ * x + bm @ x
    return x


def flops(o, i, steps):
    m, n = min(o, i), max(o, i)
    return steps * (4.0 * m * m * n + 2.0 * m * m * m)


mats = []
for (o, i), c in FAMILIES:
    m, n = min(o, i), max(o, i)
    mats.append((o, i, c, [torch.randn(m, n) for _ in range(c)]))

rows = []
for steps in (3, 5):
    for o, i, c, ms in mats:
        t = time.perf_counter()
        for x in ms:
            ns5(x, steps)
        sec = time.perf_counter() - t
        rows.append(dict(steps=steps, shape=[o, i], count=c, seconds=sec,
                         gflops=flops(o, i, steps) * c / 1e9 / sec))
        print('NS%d %dx%d x%d: %.3f s  %.1f GFLOPS' % (steps, o, i, c, sec,
                                                       rows[-1]['gflops']), flush=True)
print('RESULT ' + json.dumps(rows), flush=True)
