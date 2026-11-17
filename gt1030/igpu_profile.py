# -*- coding: utf-8 -*-
"""igpu_profile.py — 核显训练步的分段计时小工具（给 L3/L4 定位瓶颈用）。

用法（在你的 Stack 里插几行）：
    from gt1030.igpu_profile import Prof
    prof = Prof(A.queue)
    ...
    for step in range(n):
        prof.new_step()
        with prof.sec('attn'):
            ...
        with prof.sec('mlp'):
            ...
        with prof.sec('norm'):
            ...
        with prof.sec('opt'):
            ...
    prof.report()

注意：
  · `sec()` 退出时会 `queue.finish()`（否则异步时间不可信），所以绝对值偏大、**占比可信**；
  · 想测"没有某类算子会快多少"，用 `prof.ablate('attn')` 让该段变成空操作（数值会坏，只看时间）；
  · 只统计中位数以上的步（`--warmup`），避免首步编译污染。
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict


class Prof(object):
    def __init__(self, queue, warmup=2, max_steps=50):
        self.q = queue
        self.warmup = int(warmup)
        self.max_steps = int(max_steps)
        self.step = 0
        self.acc = defaultdict(list)      # name -> [每步耗时]
        self._cur = {}
        self._skip = set()

    # ---- 步骤管理 ----
    def new_step(self):
        for k, v in self._cur.items():
            if k in self.acc and len(self.acc[k]) > 0:
                self.acc[k][-1] += v          # 同一步内多次同名累加
        self._cur = {}
        self.step += 1

    # ---- 计时 ----
    class _Sec(object):
        def __init__(self, prof, name):
            self.p, self.name = prof, name

        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self.p.q.finish()
            dt = time.perf_counter() - self.t0
            if self.name in self.p._skip:
                dt = 0.0
            if len(self.p.acc[self.name]) < self.p.step:
                self.p.acc[self.name].append(0.0)
            self.p.acc[self.name][-1] += dt
            return False

    def sec(self, name):
        return Prof._Sec(self, name)

    def ablate(self, *names):
        """把指定段置空（用于"去掉它快多少"的实验）。"""
        self._skip.update(names)
        return self

    # ---- 报告 ----
    def report(self):
        tot = None
        rows = []
        for name, ts in self.acc.items():
            use = ts[self.warmup:] if len(ts) > self.warmup else ts
            if not use:
                continue
            m = statistics.median(use)
            rows.append((name, m, len(use)))
        if not rows:
            print("  [prof] 没有数据")
            return {}
        # 用各段中位数之和做分母（finish 后每段都含同步开销，不重叠）
        s = sum(m for _, m, _ in rows)
        rows.sort(key=lambda r: -r[1])
        print("  段              中位(ms)   占比")
        for name, m, n in rows:
            print("  %-14s %8.2f  %5.1f%%   (n=%d)" % (name, m * 1e3, 100 * m / s, n))
        print("  合计            %8.2f ms" % (s * 1e3))
        return {name: m for name, m, _ in rows}
