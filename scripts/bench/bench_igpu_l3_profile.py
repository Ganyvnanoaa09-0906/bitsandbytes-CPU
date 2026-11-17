# -*- coding: utf-8 -*-
"""Component-level profiling of the 4-layer L3 Stack with LEAD's igpu_profile.Prof.

Sections are disjoint: norm fwd/bwd/dw, attention fwd/bwd, MLP fwd/bwd, optimizer.
Residual/add/host-bridge time remains outside sections; total is measured separately.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gt1030.igpu_block as B  # noqa: E402
from gt1030.igpu_profile import Prof  # noqa: E402

ROWS, D, FFN = 512, 512, 1368
STACK = B.Stack(ROWS, D, FFN, 8, 64, 256, 4, seed=0, final_norm=False, per_layer=True)
rng = np.random.default_rng(0)
X = (rng.standard_normal((ROWS, D)) * 0.02).astype(np.float16)
DY = (rng.standard_normal((ROWS, D)) * 1e-6).astype(np.float16)

prof = Prof(B.A.queue, warmup=2, max_steps=5)


def wrap(obj, name, sec):
    orig = getattr(obj, name)

    def fn(*a, **k):
        with prof.sec(sec):
            return orig(*a, **k)

    setattr(obj, name, fn)


for blk in STACK.blocks:
    for nm in ("forward", "backward", "weight_grad"):
        wrap(blk.norm1, nm, "norm")
        wrap(blk.norm2, nm, "norm")
    wrap(blk.attn, "forward", "attn")
    wrap(blk.attn, "backward", "attn")
    wrap(blk.mlp, "forward", "mlp")
    wrap(blk.mlp, "backward", "mlp")
    wrap(blk, "step", "opt")

tot = []
for _ in range(7):  # 2 warmup + 5 measured
    prof.new_step()
    t0 = time.perf_counter()
    STACK.forward_np(X)
    dy = DY
    STACK.backward_np(dy)
    STACK.step(1e-5, 1, eps=1e-8 * 65536)
    B.A.queue.finish()
    tot.append(time.perf_counter() - t0)

med_total = statistics.median(tot[2:])
print("\n总步时（含 bridge/add）: %.1f ms" % (med_total * 1e3))
comps = prof.report()
for name, m in sorted(comps.items(), key=lambda kv: -kv[1]):
    print("  %-6s %7.1f ms  %5.1f%% of total" % (name, m * 1e3, 100 * m / med_total))
print(
    "  %-6s %7.1f ms  %5.1f%% of total"
    % (
        "other",
        (med_total - sum(comps.values())) * 1e3,
        100 * (med_total - sum(comps.values())) / med_total,
    )
)
