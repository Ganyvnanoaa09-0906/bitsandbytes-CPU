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


class EventProf(object):
    """OpenCL **event** 版分段计时（队列需 PROFILING_ENABLE）。

    做法：每段前后各入队一个 1 线程的哑 kernel 作为时间标记，段结束时读两个标记的
    event 时间戳算区间。与 `sec()` 的 finish 版相比：不强制串行，能看出段的真实 GPU 占用，
    也更接近"段内 kernel 之和"。

    用法：
        prof = EventProf(A.queue)
        for step in range(n):
            prof.new_step()
            with prof.sec('attn'): ...
            with prof.sec('mlp'): ...
        prof.report()
    """

    SRC = "__kernel void _mark(__global int* x){ if (get_global_id(0)==0) x[0]+=1; }"

    def __init__(self, queue, warmup=2):
        import pyopencl as cl
        import numpy as np
        from collections import defaultdict
        self.cl, self.np, self.defaultdict = cl, np, defaultdict
        self.q = queue
        self.warmup = int(warmup)
        self.step = 0
        self.prog = cl.Program(queue.context, self.SRC).build()
        self.mark_k = cl.Kernel(self.prog, "_mark")
        self.mark_buf = cl.Buffer(queue.context, cl.mem_flags.READ_WRITE, 64)
        self.pairs = []          # (step, name, t0_event, t1_event)

    def _mark(self):
        return self.mark_k(self.q, (1,), None, self.mark_buf)

    def new_step(self):
        self.step += 1

    class _Sec(object):
        def __init__(self, prof, name):
            self.p, self.name = prof, name

        def __enter__(self):
            self.t0 = self.p._mark()
            return self

        def __exit__(self, *exc):
            t1 = self.p._mark()
            self.p.pairs.append((self.p.step, self.name, self.t0, t1))
            return False

    def sec(self, name):
        return EventProf._Sec(self, name)

    def report(self):
        self.q.finish()
        acc = self.defaultdict(list)
        for step, name, t0, t1 in self.pairs:
            if step <= self.warmup:
                continue
            try:
                dur = (t1.profile.end - t0.profile.end) / 1e9     # 秒
            except Exception:
                continue
            acc[name].append(dur)
        import statistics
        rows = [(n, statistics.median(v), len(v)) for n, v in acc.items() if v]
        if not rows:
            print("  [event-prof] 没有数据（队列是否带 PROFILING_ENABLE？）")
            return {}
        s = sum(m for _, m, _ in rows)
        rows.sort(key=lambda r: -r[1])
        print("  段(event)        中位(ms)   占比")
        for n, m, k in rows:
            print("  %-14s %8.2f  %5.1f%%  (n=%d)" % (n, m * 1e3, 100 * m / s, k))
        print("  合计(可重叠)      %8.2f ms" % (s * 1e3))
        return {n: m for n, m, _ in rows}
