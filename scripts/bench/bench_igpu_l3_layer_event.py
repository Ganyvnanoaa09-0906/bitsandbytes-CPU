# -*- coding: utf-8 -*-
"""Single-layer L3 Block with EventProf: attn / mlp / norm / bridge / opt."""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gt1030.igpu_block as B  # noqa: E402
from gt1030.igpu_profile import EventProf  # noqa: E402

rows, d, ffn, nh, hd, t = 512, 512, 1368, 8, 64, 256
lc = B.LayerCtx(arena_mb=500, name="layer", profiling=True)
lc.activate()
blk = B.Block(
    rows,
    d,
    ffn,
    attn=B.SelfAttention(rows, d, nh, hd, t, bias=False, rng=np.random.default_rng(0)),
    seed=0,
    dev=lc,
)
rng = np.random.default_rng(0)
xh = (rng.standard_normal((rows, d)) * 0.02).astype(np.float16)
dyh = (rng.standard_normal((rows, d)) * 1e-6).astype(np.float16)
xd = B._upload(xh)
dyb = B._upload(dyh)
y = B._buf(rows * d * 2)
prof = EventProf(lc.queue, warmup=2)
wall = []


def add(a, b, c):
    with prof.sec("add"):
        B._K["add_half"](B.A.queue, (rows * d,), None, a, b, c, np.int32(rows * d))


for _ in range(7):
    prof.new_step()
    t0 = time.perf_counter()
    with prof.sec("bridge"):
        B.cl.enqueue_copy(B.A.queue, xd, np.ascontiguousarray(xh))
        B.cl.enqueue_copy(B.A.queue, dyb, np.ascontiguousarray(dyh))
    with prof.sec("norm"):
        h1 = blk.norm1.forward(xd)
    with prof.sec("attn"):
        a = blk.attn.forward(h1)
    add(xd, a, blk.x1)
    with prof.sec("norm"):
        h2 = blk.norm2.forward(blk.x1)
    with prof.sec("mlp"):
        blk.mlp.forward(h2)
    add(blk.x1, blk.mlp._last_y, blk.y)
    with prof.sec("bridge"):
        B._read(blk.y, (rows, d))
    with prof.sec("mlp"):
        g2 = blk.mlp.backward(dyb)
    with prof.sec("norm"):
        blk.norm2.weight_grad(g2)
        d_norm2 = blk.norm2.backward(g2)
    add(dyb, d_norm2, blk.dx1)
    with prof.sec("attn"):
        a_dx = blk.attn.backward(blk.dx1)
    with prof.sec("norm"):
        blk.norm1.weight_grad(a_dx)
        dh1 = blk.norm1.backward(a_dx)
    add(blk.dx1, dh1, blk.dx)
    with prof.sec("bridge"):
        B._read(blk.dx, (rows, d))
    with prof.sec("opt"):
        blk.step(1e-5, 1, eps=1e-8 * 65536)
    lc.queue.finish()
    wall.append(time.perf_counter() - t0)

med = statistics.median(wall[2:])
print("wall %.1f ms" % (med * 1e3))
comps = prof.report()
print(
    "segment sum %.1f ms (%.1f%% of wall)"
    % (sum(comps.values()) * 1e3, 100 * sum(comps.values()) / med)
)
print(
    "other %.1f ms (%.1f%% of wall)"
    % ((med - sum(comps.values())) * 1e3, 100 * (med - sum(comps.values())) / med)
)
prof.report_overlap()
