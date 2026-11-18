# -*- coding: utf-8 -*-
"""igpu_cpu_concurrency2.py — 并发调参：CPU 线程数 vs 与核显重叠的效果。

上一轮发现：6 线程 CPU GEMM + iGPU GEMM 并发时几乎**完全不重叠**（并发≈顺序）。
可能原因：(a) torch 6 线程占满 6 个逻辑核，OpenCL 驱动线程被饿死；
          (b) 15 W TDP 共享，CPU 与核显同时满载会互相降频。
这里扫 CPU 线程数（6/4/3/2）与是否并发，看能否找到真正的重叠收益。
"""
from __future__ import annotations

import ctypes
import json
import threading
import time

import numpy as np
import pyopencl as cl
import torch

DLL = r"D:\work\clblast_build\CLBlast-master\build\clblast.dll"
OUT = r"D:\work\research\bench_results\audio_lab"
ROW, NO = 101, 111


class G:
    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.q = cl.CommandQueue(self.ctx)
        self.bl = ctypes.CDLL(DLL)
        self.bl.CLBlastSgemm.restype = ctypes.c_int
        self.bl.CLBlastSgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        self.qp = ctypes.c_void_p(int(self.q.int_ptr))
        self.b = {}

    def gemm(self, a, b, key):
        a = np.ascontiguousarray(a, np.float32)
        b = np.ascontiguousarray(b, np.float32)
        m, k = a.shape
        _, n = b.shape
        ab = self.b.get("a") or cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, a.nbytes); self.b["a"] = ab
        bb = self.b.get("b") or cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, b.nbytes); self.b["b"] = bb
        cb = self.b.get("c") or cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, m * n * 4); self.b["c"] = cb
        cl.enqueue_copy(self.q, ab, a)
        cl.enqueue_copy(self.q, bb, b)
        rc = self.bl.CLBlastSgemm(ROW, NO, NO, m, n, k, ctypes.c_float(1.0),
                                  ctypes.c_void_p(int(ab.int_ptr)), 0, k,
                                  ctypes.c_void_p(int(bb.int_ptr)), 0, n, ctypes.c_float(0.0),
                                  ctypes.c_void_p(int(cb.int_ptr)), 0, n, self.qp, None)
        if rc:
            raise RuntimeError(rc)
        self.q.finish()


def main():
    g = G()
    n = 2048
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    At, Bt = torch.from_numpy(A), torch.from_numpy(B)
    res = {}
    for threads in (6, 4, 3, 2):
        torch.set_num_threads(threads)
        At @ Bt; g.gemm(A, B, "c")
        reps = 3
        # CPU 单独
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); At @ Bt; ts.append(time.perf_counter() - t)
        cpu_t = min(ts)
        # GPU 单独（CPU 空闲）
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); g.gemm(A, B, "c2"); ts.append(time.perf_counter() - t)
        gpu_t = min(ts)
        # 顺序
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); At @ Bt; g.gemm(A, B, "c3"); ts.append(time.perf_counter() - t)
        seq_t = min(ts)
        # 并发
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            th = threading.Thread(target=lambda: g.gemm(A, B, "c4"))
            th.start()
            At @ Bt
            th.join()
            ts.append(time.perf_counter() - t0)
        conc_t = min(ts)
        key = f"threads{threads}"
        res[key] = dict(cpu_ms=round(cpu_t * 1e3, 1), igpu_ms=round(gpu_t * 1e3, 1),
                        seq_ms=round(seq_t * 1e3, 1), conc_ms=round(conc_t * 1e3, 1),
                        overlap_gain=round(seq_t / conc_t, 2))
        print(f"  CPU {threads} 线程: CPU {cpu_t*1e3:6.1f} | iGPU {gpu_t*1e3:6.1f} | 顺序 {seq_t*1e3:6.1f} | "
              f"并发 {conc_t*1e3:6.1f} | 重叠收益 {seq_t/conc_t:4.2f}×", flush=True)
    torch.set_num_threads(6)
    with open(rf"{OUT}\cpu_igpu_concurrency2.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
