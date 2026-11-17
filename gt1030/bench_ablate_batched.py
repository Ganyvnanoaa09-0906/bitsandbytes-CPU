# -*- coding: utf-8 -*-
"""bench_ablate_batched.py — 消融：把 attention 反向的 StridedBatched GEMM 置空，看步时掉多少。

用途：判断"核显一步 1.23 s"里 attention 反向占多大（数值会坏，只看时间）。
用法：python gt1030/bench_ablate_batched.py
"""
from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402


def run(model, toks, rng, steps, label):
    ts = []
    pos = 0
    order = rng.permutation(len(toks))
    for st in range(1, steps + 1):
        if pos + MM.B > len(order):
            order = rng.permutation(len(toks))
            pos = 0
        idx = order[pos:pos + MM.B]
        pos += MM.B
        tokens = np.ascontiguousarray(toks[idx].astype(np.int64))
        t0 = time.perf_counter()
        model.forward(tokens)
        model.backward(tokens)
        model.opt_step(5e-7, st)
        dt = time.perf_counter() - t0
        if st > 1:
            ts.append(dt)
    print(f"  {label}: 中位步时 {statistics.median(ts):.3f}s  (n={len(ts)})", flush=True)
    return statistics.median(ts)


def main() -> int:
    toks = torch.load(r"F:\weibo_cos\tokens_32x32_weibo.pt", map_location="cpu",
                      weights_only=False)
    toks = toks.numpy() if torch.is_tensor(toks) else np.asarray(toks)
    toks = np.ascontiguousarray(toks[:, : MM.T])

    model = MM.Model(r"F:\weibo_cos\_from76000\ckpt_76400.pt")
    base = run(model, toks, np.random.default_rng(0), 6, "基线")

    orig = MM.A.hgemm_batched
    MM.A.hgemm_batched = lambda *a, **k: None
    try:
        ablat = run(model, toks, np.random.default_rng(1), 6, "去掉 batched GEMM")
    finally:
        MM.A.hgemm_batched = orig
    print(f"  ⇒ batched GEMM 占每步 {base - ablat:.3f}s ({100*(base-ablat)/base:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
