# -*- coding: utf-8 -*-
"""Ablation timing for 4-layer L3: disable one kernel class at a time (numbers break)."""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gt1030.igpu_block as B  # noqa: E402
import gt1030.ar_v2_multi as MM  # noqa: E402
import gt1030.igpu_attn as IA  # noqa: E402

ROWS, D, FFN = 512, 512, 1368
STACK = B.Stack(ROWS, D, FFN, 8, 64, 256, 4, seed=0, final_norm=False, per_layer=True)
rng = np.random.default_rng(0)
X = (rng.standard_normal((ROWS, D)) * 0.02).astype(np.float16)
DY = (rng.standard_normal((ROWS, D)) * 1e-6).astype(np.float16)
G = 65536.0


def one_step():
    STACK.forward_np(X)
    STACK.backward_np(DY)
    STACK.step(1e-5, 1, eps=1e-8 * G)
    B.A.queue.finish()


def step_time(n=5):
    for _ in range(2):
        one_step()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        one_step()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


class Abl:
    def __init__(self):
        self.saved = []

    def module(self, mod, names):
        for n in names:
            self.saved.append((mod, n, getattr(mod, n)))
            setattr(mod, n, lambda *a, **k: None)

    def dict(self, d, prefixes):
        for k in list(d):
            if any(k.startswith(p) for p in prefixes):
                self.saved.append((d, k, d[k]))
                d[k] = lambda *a, **k2: None

    def restore(self):
        for obj, key, fn in reversed(self.saved):
            if isinstance(obj, dict):
                obj[key] = fn
            else:
                setattr(obj, key, fn)
        self.saved = []


base = step_time()
print("base %.1f ms" % (base * 1e3))

specs = [
    ("gemm_all", lambda ab: ab.module(MM.L, ("hgemm_nn", "hgemm_nt", "hgemm_tn"))),
    ("softmax", lambda ab: [ab.dict(d, ("softmax",)) for d in IA._K.values()]),
    (
        "norm",
        lambda ab: [ab.dict(blk.dev.norm_K, ("rmsnorm_",)) for blk in STACK.blocks],
    ),
    (
        "reshape",
        lambda ab: [
            ab.dict(
                blk.dev.K, ("split_qkv", "merge_qkv", "heads_to_tok", "tok_to_heads")
            )
            for blk in STACK.blocks
        ],
    ),
    ("opt", lambda ab: setattr(STACK, "step", lambda *a, **k: None)),
]

for name, apply in specs:
    ab = Abl()
    if name == "opt":
        ab.saved.append((STACK, "step", STACK.step))
        STACK.step = lambda *a, **k: None
    else:
        apply(ab)
    t = step_time()
    ab.restore()
    print(
        "  %-9s %.1f ms  saves %+5.1f ms (%+5.1f%%)"
        % (name, t * 1e3, (base - t) * 1e3, 100 * (base - t) / base)
    )
