# -*- coding: utf-8 -*-
"""bench_ablate_groups.py — 分组消融：找出核显一步 1.24 s 里剩下的 0.7 s 在哪。

组：gemm(L.hgemm_*)、attn_matmul(K:sgemm_*)、softmax(K:softmax*)、rms(K:rms*)、
    transpose/copy(K:transpose*)、adamw(K:adamw*)
每组置空看步时下降多少（数值会坏，只看时间）。
"""
from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402

TOKS = torch.load(r"F:\weibo_cos\tokens_32x32_weibo.pt", map_location="cpu", weights_only=False)
TOKS = TOKS.numpy() if torch.is_tensor(TOKS) else np.asarray(TOKS)
TOKS = np.ascontiguousarray(TOKS[:, : MM.T])


def step_time(model, seed, steps=6):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(TOKS))
    pos, ts = 0, []
    for st in range(1, steps + 1):
        idx = order[pos:pos + MM.B]
        pos += MM.B
        tokens = np.ascontiguousarray(TOKS[idx].astype(np.int64))
        t0 = time.perf_counter()
        model.forward(tokens)
        model.backward(tokens)
        model.opt_step(5e-7, st)
        dt = time.perf_counter() - t0
        if st > 1:
            ts.append(dt)
    return statistics.median(ts)


class Ablator:
    def __init__(self, model):
        self.model = model
        self.saved = []

    def by_prefix(self, prefixes, attr="K"):
        tgt = self.model.dev.K if attr == "K" else None
        for name in list(tgt):
            if any(name.startswith(p) for p in prefixes):
                self.saved.append((name, tgt[name]))
                tgt[name] = (lambda *a, **k: None)
        return [n for n, _ in self.saved if any(n.startswith(p) for p in prefixes)]

    def by_module(self, names):
        for nm in names:
            if hasattr(MM.L, nm):
                fn = getattr(MM.L, nm)
                self.saved.append((f"L.{nm}", fn))
                setattr(MM.L, nm, (lambda *a, **k: None))
        return [f"L.{nm}" for nm in names]

    def restore(self):
        for name, fn in self.saved:
            if name.startswith("L."):
                setattr(MM.L, name[2:], fn)
            else:
                self.model.dev.K[name] = fn
        self.saved = []


def main() -> int:
    model = MM.Model(r"F:\weibo_cos\_from76000\ckpt_76400.pt")
    base = step_time(model, 0)
    print(f"  基线中位步时 {base:.3f}s", flush=True)
    groups = [
        ("gemm_all", ["hgemm_nn", "hgemm_nt", "hgemm_tn"]),
        ("gemm_nn_only", ["hgemm_nn"]),
        ("attn_matmul", ["sgemm_nn_bt", "sgemm_nt64", "sgemm_nt"]),
        ("softmax", ["softmax"]),
        ("rms", ["rms_"]),
        ("adamw", ["adamw"]),
        ("transpose", ["transpose"]),
        ("split_merge", ["split_", "merge_"]),
        ("qk_norm", ["qk_norm"]),
        ("scale", ["scale_"]),
        ("col_sum", ["col_sum"]),
        ("allsmall", ["split_", "merge_", "qk_norm", "scale_", "col_sum", "sumsq", "rms_dw"]),
    ]
    for label, spec in groups:
        ab = Ablator(model)
        if label.startswith("gemm"):
            ab.by_module(spec)
        else:
            ab.by_prefix(spec)
        t = step_time(model, 1)
        ab.restore()
        print(f"  {label:12s} → {t:.3f}s  省 {base-t:+.3f}s ({100*(base-t)/base:+.0f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
