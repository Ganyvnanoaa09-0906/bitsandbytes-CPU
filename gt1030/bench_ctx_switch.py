# -*- coding: utf-8 -*-
"""bench_ctx_switch.py — 量化"per-layer context 切换"的代价（解释 L3 的 other 段）。

背景：L3 的 4 层 stack 每层一个 DevCtx；`activate()` 是纯 host 操作，
但**每次切换后第一批 kernel 的驱动开销**可能很大（CLBlast 还要重绑 QP）。

用法：
    python gt1030/bench_ctx_switch.py            # 默认 4 层
    python gt1030/bench_ctx_switch.py --layers 2 --buffers 40
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402
import gt1030.igpu_ctx as CTX  # noqa: E402

A = MM.A

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void touch(__global half* x, int n) {
    int i = get_global_id(0); if (i >= n) return;
    x[i] = (half)((float)x[i] + 1.0f);
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--buffers", type=int, default=40, help="每层预分配多少 buffer")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()

    gang = CTX.LayerGang(n_layers=a.layers, arena_mb=200, prefix="L")
    ks, bufs, progs = [], [], []
    for d in gang.layers:
        d.activate()
        progs.append(d.program(SRC, ["touch"]))
        ks.append(progs[-1]["touch"])
        bufs.append([d.buf(64 * 1024) for _ in range(a.buffers)])   # 64KB × N

    # 1) 纯 activate 切换（不做任何 kernel）
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        for d in gang.layers:
            d.activate()
        ts.append(time.perf_counter() - t0)
    print("  纯 activate × %d 层: 中位 %.3f ms" % (a.layers, statistics.median(ts) * 1e3))

    # 2) 切换 + 每层一次 kernel（模拟"切完立刻 launch"）
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        for li, d in enumerate(gang.layers):
            d.activate()
            ks[li](d.q, (64 * 1024 // 2,), None, bufs[li][0], np.int32(64 * 1024 // 2))
            d.q.finish()
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    print("  切换 + 每层 1 kernel: 中位 %.3f ms（%.3f ms/层）" % (med * 1e3, med * 1e3 / a.layers))

    # 3) 单 context 内同样 kernel（对照：没有切换）
    one = CTX.DevCtx(arena_mb=200, name="single")
    one.activate()
    k1 = one.program(SRC, ["touch"])["touch"]
    bs = [one.buf(64 * 1024) for _ in range(a.buffers)]
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        for _l in range(a.layers):
            k1(one.q, (64 * 1024 // 2,), None, bs[0], np.int32(64 * 1024 // 2))
            one.q.finish()
        ts.append(time.perf_counter() - t0)
    med1 = statistics.median(ts)
    print("  单 context × %d 次 kernel: 中位 %.3f ms（对照）" % (a.layers, med1 * 1e3))
    print("  ⇒ 切换开销 ≈ %.3f ms/步（若这个数与 L3 的 other 同量级，其他段就找到了）"
          % ((med - med1) * 1e3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
