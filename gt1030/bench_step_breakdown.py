# -*- coding: utf-8 -*-
"""bench_step_breakdown.py — 核显常驻训练一步的时间/FLOPs 分解（对标 1030 用）。

目标（用户定义）：
  · 基准 = GTX 1030 fp32 纯理论 1.128 TFLOPS；核显 fp16 峰值 2.30 TFLOPS（gfx902 6CU@1.5GHz）
  · 模型 = 30.5M AR，T=1024，B=2 ⇒ 2048 token/步 ⇒ 375 GFLOP/步（6·N·tokens）
  · 1030 理论步时 = 0.333 s；当前核显端到端 ~1.45 s ⇒ 需要 4.4× 才能"追上 1030 理论"

本脚本把一步拆成 forward / backward / optimizer / 其它，并折算各自的 GFLOPS，
以便判断该优化哪一段，而不是盲目调 GEMM。

用法：
  python gt1030/bench_step_breakdown.py --ckpt F:\\weibo_cos\\_from76000\\ckpt_76400.pt --steps 20
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402

TARGET_1030_FP32_TFLOPS = 1.128
IGPU_FP16_TFLOPS = 2.30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"F:\weibo_cos\_from76000\ckpt_76400.pt")
    ap.add_argument("--tokens", default=r"F:\weibo_cos\tokens_32x32_weibo.pt")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-7)
    a = ap.parse_args()

    os.environ.setdefault("AR_DYN_SCALE", "1")
    os.environ.setdefault("AR_DYN_INIT", "1024")
    os.environ.setdefault("AR_ARENA_MB", "3100")

    toks = torch.load(a.tokens, map_location="cpu", weights_only=False)
    toks = toks.numpy() if torch.is_tensor(toks) else np.asarray(toks)
    toks = np.ascontiguousarray(toks[:, : MM.T])

    model = MM.Model(a.ckpt)
    n_img = len(toks)
    rng = np.random.default_rng(0)
    order = rng.permutation(n_img)
    pos = 0

    # FLOPs 口径：6·N·tokens（fwd 2 + bwd 4），attention 另算
    N = sum(p.n for p in (model.head.W,)) if hasattr(model.head.W, "n") else 0
    # 参数总量从 ckpt 里取更可靠
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    N = sum(v.numel() for v in ck["model"].values() if hasattr(v, "numel"))
    tokens_per_step = MM.B * MM.T
    gflop_step = 6.0 * N * tokens_per_step / 1e9

    rows = []
    for st in range(1, a.warmup + a.steps + 1):
        if pos + MM.B > len(order):
            order = rng.permutation(n_img)
            pos = 0
        idx = order[pos:pos + MM.B]
        pos += MM.B
        tokens = np.ascontiguousarray(toks[idx].astype(np.int64))

        t0 = time.perf_counter()
        loss = model.forward(tokens)
        t1 = time.perf_counter()
        model.backward(tokens)
        t2 = time.perf_counter()
        gn = model.opt_step(a.lr, st)
        t3 = time.perf_counter()
        if st > a.warmup:
            rows.append({"fwd": t1 - t0, "bwd": t2 - t1, "opt": t3 - t2,
                         "total": t3 - t0, "loss": float(loss)})

    def med(k):
        return statistics.median(r[k] for r in rows)

    total = med("total")
    fwd, bwd, opt = med("fwd"), med("bwd"), med("opt")
    eff = gflop_step / total if total > 0 else 0.0
    rep = {
        "schema_version": 1,
        "kind": "step_breakdown",
        "ckpt": os.path.basename(a.ckpt),
        "batch": int(MM.B), "T": int(MM.T), "tokens_per_step": int(tokens_per_step),
        "params_M": round(N / 1e6, 2),
        "gflop_per_step": round(gflop_step, 1),
        "median_s": {"fwd": round(fwd, 4), "bwd": round(bwd, 4), "opt": round(opt, 4),
                     "total": round(total, 4)},
        "share": {"fwd": round(fwd / total, 3), "bwd": round(bwd / total, 3),
                  "opt": round(opt / total, 3)},
        "effective_gflops": round(eff, 1),
        "pct_of_fp16_peak": round(100 * eff / (IGPU_FP16_TFLOPS * 1000), 1),
        "target_1030_s": round(gflop_step / (TARGET_1030_FP32_TFLOPS * 1000), 3),
        "gap_to_1030": round(total / (gflop_step / (TARGET_1030_FP32_TFLOPS * 1000)), 2),
        "steps_measured": len(rows),
        "loss_last": rows[-1]["loss"] if rows else None,
    }
    print("  " + json.dumps(rep, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
