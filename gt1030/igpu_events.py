# -*- coding: utf-8 -*-
"""igpu_events.py — **CLBlast GEMM 的真实 kernel event 时间戳**（区分"段内 host"与"GPU 计算"）。

问题：`igpu_profile.EventProf` 用哑 kernel 标记夹区间，区间里混着 host 提交与 GPU 空闲，
无法回答"这一段到底有多少是 GPU 在算"。

做法：CLBlast 的 C API 最后一个参数是 `cl_event*`；我们在 `ar_v2_multi` 里传的是 `None`。
本模块改成传 `ctypes.byref(evp)`，再用 OpenCL 的 `clGetEventProfilingInfo` 读回
START(0x1282)/END(0x1283) 两个纳秒时间戳 —— **这就是纯 GPU 执行时间**，不含 host。

前提：队列必须开 `PROFILING_ENABLE`（用 `DevCtx(profiling=True)`）。

用法：
    from gt1030.igpu_events import GemmEv
    ge = GemmEv()                          # 首次调用会加载 OpenCL.dll
    ev = ge.hgemm_tn(dy, x, dW, N, K, M)   # 语义与 L.hgemm_tn 完全一致，返回 event 句柄
    ...                                    # 可以连续提交多个
    ge.q.finish()
    print(ge.ms(ev))                       # 该 GEMM 的纯 GPU 毫秒
    ge.close(ev)                           # 释放 event（务必，否则泄漏）

    # 或者一次性批量：
    with ge.section('qkv_b'):
        ge.hgemm_tn(...); ge.hgemm_nn(...)
    ge.report()                            # 按 section 汇总纯 GPU 时间
"""
from __future__ import annotations

import ctypes
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import gt1030.ar_v2_multi as MM  # noqa: E402

A = MM.A

CL_PROFILING_COMMAND_START = 0x1282
CL_PROFILING_COMMAND_END = 0x1283
CL_PROFILING_COMMAND_QUEUED = 0x1280
CL_PROFILING_COMMAND_SUBMIT = 0x1281


def _load_opencl():
    last = None
    for name in ("OpenCL.dll", "C:/Windows/System32/OpenCL.dll"):
        try:
            dll = ctypes.WinDLL(name)
            break
        except Exception as e:      # pragma: no cover
            last = e
            dll = None
    if dll is None:
        raise RuntimeError("加载 OpenCL.dll 失败: %r" % (last,))
    dll.clGetEventProfilingInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                            ctypes.c_size_t, ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_size_t)]
    dll.clGetEventProfilingInfo.restype = ctypes.c_int
    dll.clReleaseEvent.argtypes = [ctypes.c_void_p]
    dll.clReleaseEvent.restype = ctypes.c_int
    return dll


class GemmEv(object):
    """带 event 的 CLBlast 包装；语义与原 L.hgemm_* 一致。

    ⚠️ **实测结论（2026-11-17，gfx902 + CLBlast）——两条都会误导，别用这种口径**：
    1. CLBlast 返回的 `cl_event` **不是 GEMM 主 kernel 的事件**：
       512×1536×512 的 hgemm_nt，event 法 **0.199 ms** vs 实际 **1.72 ms**（无 event 时）→ 差 8.6×；
    2. **传 event 指针本身会让 GEMM 变慢**：同一形状
       带 event 指针 **3.04 ms** vs 不带 **1.72 ms**（**1.8× 开销**），
       而 profiling 队列本身几乎不花钱（3.04 vs 3.10 ms，差 2%）。

    ⇒ 结论：**event 指针既不准又拖慢**，要拆 GEMM 请用本模块的 `amortized()`
    （同一算子连续提交 N 次再除以 N，host 提交 ~0.017 ms/次摊薄到 1% 以内，且不拖慢）。

    本类保留 event 通道，仅用于：① 拿到提交顺序；② 对**自己写的 kernel**（pyopencl 返回
    event 的那类）做精确计时；③ 复核 CLBlast 内部是否只覆盖了部分 kernel。
    """

    def __init__(self):
        self.ocl = _load_opencl()
        self._sec = None
        self.acc = {}          # section -> [ms]
        self.pending = []      # [(section, ev_ptr)]

    # ---------------- 三个 GEMM（签名与 MM.L 相同） ----------------
    def _call(self, ta, tb, a_dev, b_dev, c_dev, m, n, k, lda, ldb, ldc, alpha=None):
        evp = ctypes.c_void_p()
        rc = A.bl.CLBlastHgemm(
            A.ROW, ta, tb, m, n, k, A.ONE if alpha is None else alpha,
            ctypes.c_void_p(int(a_dev.int_ptr)), 0, lda,
            ctypes.c_void_p(int(b_dev.int_ptr)), 0, ldb,
            A.ZERO,
            ctypes.c_void_p(int(c_dev.int_ptr)), 0, ldc,
            ctypes.byref(A.QP), ctypes.byref(evp))
        if rc != 0:
            raise RuntimeError("CLBlastHgemm rc=%d m%d n%d k%d" % (rc, m, n, k))
        if evp.value:
            self.pending.append((self._sec or "", evp))
        return evp

    def hgemm_nt(self, x, w, y, m, n, k):        # y[m,n] = x[m,k] @ w[n,k]^T
        return self._call(A.NO, A.YES, x, w, y, m, n, k, k, k, n)

    def hgemm_tn(self, dy, x, dw, n, k, m):      # dw[n,k] = dy[m,n]^T @ x[m,k]
        return self._call(A.YES, A.NO, dy, x, dw, n, k, m, n, k, k)

    def hgemm_nn(self, dy, w, dx, m, k, n):      # dx[m,k] = dy[m,n] @ w[n,k]
        return self._call(A.NO, A.NO, dy, w, dx, m, k, n, n, k, k)

    # ---------------- event 读取 ----------------
    def _info(self, ptr, param):
        v = ctypes.c_ulonglong(0)
        rc = self.ocl.clGetEventProfilingInfo(ctypes.c_void_p(ptr), ctypes.c_uint(param),
                                              8, ctypes.byref(v), None)
        return (v.value if rc == 0 else None)

    def ms(self, evp):
        """纯 GPU 毫秒（START→END）。"""
        if not evp or not evp.value:
            return float("nan")
        t0 = self._info(evp.value, CL_PROFILING_COMMAND_START)
        t1 = self._info(evp.value, CL_PROFILING_COMMAND_END)
        if t0 is None or t1 is None:
            return float("nan")
        return (t1 - t0) / 1e6

    def close(self, evp):
        if evp and evp.value:
            self.ocl.clReleaseEvent(ctypes.c_void_p(evp.value))

    # ---------------- 分段收集 ----------------
    class _Sec(object):
        def __init__(self, ge, name):
            self.ge, self.name, self.prev = ge, name, None

        def __enter__(self):
            self.prev = self.ge._sec
            self.ge._sec = self.name
            return self

        def __exit__(self, *exc):
            self.ge._sec = self.prev
            return False

    def section(self, name):
        return GemmEv._Sec(self, name)

    def flush(self):
        """等待 GPU 完成并把 pending 的耗时按 section 汇总（可重复调用）。"""
        A.queue.finish()
        for sec, evp in self.pending:
            t = self.ms(evp)
            if t == t:
                self.acc.setdefault(sec, []).append(t)
            self.close(evp)
        self.pending = []

    def report(self):
        self.flush()
        import statistics
        if not self.acc:
            print("  [gemm-ev] 没有数据")
            return {}
        total = sum(statistics.median(v) for v in self.acc.values())
        print("  section(GEMM)     中位(ms)   占 GEMM")
        out = {}
        for name, v in sorted(self.acc.items(), key=lambda kv: -statistics.median(kv[1])):
            m = statistics.median(v)
            out[name] = m
            print("  %-16s %8.3f   %5.1f%%  (n=%d)" % (name, m, 100 * m / total, len(v)))
        print("  GEMM 合计        %8.3f ms" % total)
        return out


def amortized(fn, n=20, queue=None):
    """把同一算子连续提交 n 次，返回**摊薄 host 开销后的单次毫秒数**。

    这是拆 GEMM 的可靠方法：host 提交 ~0.003–0.017 ms/次，n=20 时误差 <1%；
    比 CLBlast 的 event（只覆盖部分 kernel）和单次 wall（受抖动影响）都稳。
    """
    import time
    q = queue if queue is not None else A.queue
    fn()
    q.finish()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    q.finish()
    return (time.perf_counter() - t0) / n * 1e3


def _demo():
    """冒烟：对比 event / wall / amortized 三种口径，说明为什么用 amortized。"""
    import numpy as np
    import gt1030.igpu_ctx as CTX
    d = CTX.DevCtx(arena_mb=600, name="gemm-ev", profiling=True)
    d.activate()
    M, K, N = 512, 512, 1536
    x = d.up(np.zeros((M, K), np.float16))
    w = d.up(np.zeros((N, K), np.float16))
    y = d.buf(M * N * 2)
    ge = GemmEv()
    t_ev = []
    for _ in range(10):
        ev = ge.hgemm_nt(x, w, y, M, N, K)
        ge.close_after = None
        d.q.finish()
        t_ev.append(ge.ms(ev))
        ge.close(ev)
    t_am = amortized(lambda: ge.hgemm_nt(x, w, y, M, N, K), n=20, queue=d.q)
    flops = 2 * M * N * K
    print("  hgemm_nt %dx%dx%d = %.3f GFLOP" % (M, N, K, flops / 1e9))
    print("  event 中位      : %.3f ms  (不可信，见类注释)" % sorted(t_ev)[5])
    print("  amortized(N=20) : %.3f ms  → %.0f GFLOPS" % (t_am, flops / t_am / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
