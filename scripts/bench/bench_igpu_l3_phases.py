# -*- coding: utf-8 -*-
"""Phase-level timing for the 4-layer L3 Stack (fwd / bwd / optimizer)."""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gt1030.igpu_block as B  # noqa: E402

ROWS, D, FFN = 512, 512, 1368
STACK = B.Stack(ROWS, D, FFN, 8, 64, 256, 4, seed=0, final_norm=False, per_layer=True)
rng = np.random.default_rng(0)
X = (rng.standard_normal((ROWS, D)) * 0.02).astype(np.float16)
DY = (rng.standard_normal((ROWS, D)) * 1e-6).astype(np.float16)

for _ in range(2):
    STACK.forward_np(X)
    STACK.backward_np(DY)
    STACK.step(1e-5, 1, eps=1e-8 * 65536)
B.A.queue.finish()

tf, tb, ts = [], [], []
for _ in range(10):
    t0 = time.perf_counter()
    STACK.forward_np(X)
    t1 = time.perf_counter()
    STACK.backward_np(DY)
    t2 = time.perf_counter()
    STACK.step(1e-5, 1, eps=1e-8 * 65536)
    B.A.queue.finish()
    t3 = time.perf_counter()
    tf.append(t1 - t0)
    tb.append(t2 - t1)
    ts.append(t3 - t2)


def med(a):
    return statistics.median(a) * 1e3


print(
    "4L L3 phase: fwd %.1f ms | bwd %.1f ms | opt %.1f ms | total %.1f ms"
    % (med(tf), med(tb), med(ts), med(tf) + med(tb) + med(ts))
)
