# -*- coding: utf-8 -*-
"""bench_kernel_attribution.py — 给核显训练的一步做 kernel 级时间归属（带 finish，只为相对占比）。

做法：把 ar_v2_multi 的 K[name] 内核、L.hgemm_* 与 A.hgemm_batched 包一层计时，
每次调用后 queue.finish()（会放大绝对值，但占比可信），跑 1 步后按累计时间排序。

用法：python gt1030/bench_kernel_attribution.py
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402

ACC = defaultdict(lambda: [0.0, 0])   # name -> [秒, 次数]


def wrap(fn, name, finisher):
    def inner(*a, **kw):
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        finisher()
        ACC[name][0] += time.perf_counter() - t0
        ACC[name][1] += 1
        return r
    return inner


def main() -> int:
    fin = MM.A.queue.finish
    toks = torch.load(r"F:\weibo_cos\tokens_32x32_weibo.pt", map_location="cpu",
                      weights_only=False)
    toks = toks.numpy() if torch.is_tensor(toks) else np.asarray(toks)
    toks = np.ascontiguousarray(toks[:, : MM.T])
    model = MM.Model(r"F:\weibo_cos\_from76000\ckpt_76400.pt")

    # 1) 逐 kernel 包装（内核在 Dev 实例上）
    for k, v in list(model.dev.K.items()):
        if callable(v):
            model.dev.K[k] = wrap(v, f"K:{k}", fin)
    # 2) GEMM 包装
    for nm in ("hgemm_nn", "hgemm_nt", "hgemm_tn", "hgemm_batched"):
        if hasattr(MM.L, nm):
            setattr(MM.L, nm, wrap(getattr(MM.L, nm), f"L:{nm}", fin))
    if hasattr(MM.A, "hgemm_batched"):
        MM.A.hgemm_batched = wrap(MM.A.hgemm_batched, "A:hgemm_batched", fin)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(toks))[: MM.B]
    tokens = np.ascontiguousarray(toks[idx].astype(np.int64))

    t0 = time.perf_counter()
    loss = model.forward(tokens)
    t1 = time.perf_counter()
    model.backward(tokens)
    t2 = time.perf_counter()
    model.opt_step(5e-7, 1)
    t3 = time.perf_counter()
    total = t3 - t0
    print(f"  总步时 {total:.3f}s | fwd {t1-t0:.3f} bwd {t2-t1:.3f} opt {t3-t2:.3f}")
    accounted = sum(v[0] for v in ACC.values())
    print(f"  已归属 {accounted:.3f}s ({100*accounted/total:.0f}%) | 其余(host/launch/copy) "
          f"{total-accounted:.3f}s")
    print("  ---- Top 18 ----")
    for name, (s, n) in sorted(ACC.items(), key=lambda kv: -kv[1][0])[:18]:
        print(f"    {name:26s} {s*1e3:8.1f} ms  x{n:<5d} {100*s/total:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
