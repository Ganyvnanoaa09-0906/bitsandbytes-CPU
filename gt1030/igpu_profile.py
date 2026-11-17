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
            import time
            self.h0 = time.perf_counter()
            self.t0 = self.p._mark()
            return self

        def __exit__(self, *exc):
            t1 = self.p._mark()
            import time
            self.h1 = time.perf_counter()
            self.p.pairs.append((self.p.step, self.name, self.t0, t1, self.h0, self.h1))
            return False

    def sec(self, name):
        return EventProf._Sec(self, name)

    def report(self):
        self.q.finish()
        acc = self.defaultdict(list)
        for step, name, t0, t1, _h0, _h1 in self.pairs:
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


    # ---------------------------------------------------------------- 时间线/gap 归因
    def report_overlap(self, top=8):
        """按 event 时间戳重建时间线：GPU 忙区间、空闲 gap、每段 host 提交耗时。

        这是回答"段和只有 wall 的 48%"的工具：
          · gpu  = 各段 [start, end] 的并集（真正在算的时间）
          · gap  = wall − gpu（host 提交/驱动/同步间隙）
          · host = 每段在 host 侧停留的时间（≈ 提交这批 kernel 的 cost）
        """
        self.q.finish()
        rows = []
        for step, name, t0, t1, h0, h1 in self.pairs:
            if step <= self.warmup:
                continue
            try:
                s = t0.profile.end / 1e9
                e = t1.profile.end / 1e9
            except Exception:
                continue
            rows.append((s, e, name, h0, h1))
        if not rows:
            print("  [overlap] 没有数据")
            return {}
        rows.sort()

        # GPU 忙区间并集
        busy = 0.0
        cur_s, cur_e = rows[0][0], rows[0][1]
        gaps = []
        for s, e, name, _, _ in rows[1:]:
            if s > cur_e:
                busy += cur_e - cur_s
                gaps.append((cur_e, s))
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        busy += cur_e - cur_s
        wall = rows[-1][1] - rows[0][0]

        # 每段 host 提交成本
        host = {}
        for s, e, name, h0, h1 in rows:
            host.setdefault(name, []).append(h1 - h0)
        import statistics
        host_med = {k: statistics.median(v) for k, v in host.items()}
        total_host = sum(host_med.values())

        print("  时间线: wall %.2f ms | GPU 忙 %.2f ms (%.0f%%) | gap %.2f ms (%.0f%%)"
              % (wall * 1e3, busy * 1e3, 100 * busy / wall, (wall - busy) * 1e3,
                 100 * (wall - busy) / wall))
        print("  host 提交合计（各段 host 停留之和）: %.2f ms" % (total_host * 1e3))
        print("  段                 GPU(ms)  host(ms)")
        for s, e, name, h0, h1 in rows:
            print("  %-18s %7.2f %8.2f" % (name, (e - s) * 1e3, (h1 - h0) * 1e3))
        if gaps:
            gaps.sort(key=lambda g: -(g[1] - g[0]))
            print("  最大的 %d 个 GPU 空闲 gap（前一段结束 -> 后一段开始）:" % min(top, len(gaps)))
            for g0, g1 in gaps[:top]:
                # 找 gap 前后的段名
                before = [r[2] for r in rows if r[1] <= g0 + 1e-12]
                after = [r[2] for r in rows if r[0] >= g1 - 1e-12]
                print("    %7.2f ms   after %-16s before %s"
                      % ((g1 - g0) * 1e3, before[-1] if before else '?',
                         after[0] if after else '?'))
        return {"wall": wall, "busy": busy, "gap": wall - busy, "host": total_host}
