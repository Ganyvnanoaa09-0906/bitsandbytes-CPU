# -*- coding: utf-8 -*-
"""igpu_cpu_concurrency.py — 核显与 CPU 并发/异步：到底能不能叠加算力？

测三件事：
  1) 单引擎吞吐：CPU(torch 6 线程) vs iGPU(CLBlast) 的 GEMM 时间；
  2) 顺序执行（先 CPU 后 iGPU）总时间；
  3) 并发执行（iGPU 在子线程异步跑，CPU 主线程同时跑另一个 GEMM）总时间，
     再看"比最好的单引擎快多少"（理想 = 两者相加后减半，实际受共享 DDR4 带宽限制）。

另外测「按行切分」：把一个 GEMM 的若干行分给 CPU、其余给 iGPU，看总时间是否优于单引擎。
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
ROW, NO, YES = 101, 111, 112
torch.set_num_threads(6)


class G:
    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.q = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
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
        self.pool = {}

    def buf(self, key, nbytes):
        b = self.pool.get(key)
        if b is None or b.size < nbytes:
            b = cl.Buffer(self.ctx, self.mf.READ_WRITE, nbytes)
            self.pool[key] = b
        return b

    def gemm_async(self, a_np, b_np, c_key):
        a = np.ascontiguousarray(a_np, np.float32)
        b = np.ascontiguousarray(b_np, np.float32)
        m, k = a.shape
        k2, n = b.shape
        ab = self.buf("a", a.nbytes)
        bb = self.buf("b", b.nbytes)
        cb = self.buf(c_key, m * n * 4)
        cl.enqueue_copy(self.q, ab, a)
        cl.enqueue_copy(self.q, bb, b)
        rc = self.bl.CLBlastSgemm(ROW, NO, NO, m, n, k, ctypes.c_float(1.0),
                                  ctypes.c_void_p(int(ab.int_ptr)), 0, k,
                                  ctypes.c_void_p(int(bb.int_ptr)), 0, n, ctypes.c_float(0.0),
                                  ctypes.c_void_p(int(cb.int_ptr)), 0, n, self.qp, None)
        if rc:
            raise RuntimeError(f"Sgemm rc={rc}")
        return cb, (m, n)

    def finish_and_get(self, cb, shape):
        self.q.finish()
        out = np.empty(shape, np.float32)
        cl.enqueue_copy(self.q, out, cb).wait()
        return out


def main():
    g = G()
    res = {}
    for n in (1024, 2048):
        A = np.random.rand(n, n).astype(np.float32)
        B = np.random.rand(n, n).astype(np.float32)
        At, Bt = torch.from_numpy(A), torch.from_numpy(B)
        flops = 2 * n ** 3

        def cpu_run():
            return At @ Bt

        def igpu_run():
            cb, shp = g.gemm_async(A, B, "c")
            return g.finish_and_get(cb, shp)

        # 预热
        cpu_run(); igpu_run()
        reps = 3
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); cpu_run(); ts.append(time.perf_counter() - t)
        cpu_t = min(ts)
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); igpu_run(); ts.append(time.perf_counter() - t)
        gpu_t = min(ts)

        # 顺序：两个 GEMM（一个 CPU、一个 iGPU）
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); cpu_run(); igpu_run(); ts.append(time.perf_counter() - t)
        seq_t = min(ts)

        # 并发：iGPU 在子线程，CPU 主线程同时跑
        result = {}
        def bg():
            result["t0"] = time.perf_counter()
            igpu_run()

        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            th = threading.Thread(target=bg)
            th.start()
            cpu_run()
            th.join()
            ts.append(time.perf_counter() - t0)
        conc_t = min(ts)

        # 按行切分：CPU 跑前 r 行，iGPU 跑其余（并发）
        split_res = {}
        for frac in (0.3, 0.5, 0.65, 0.8):
            m_cpu = max(1, int(n * frac))
            a_cpu = torch.from_numpy(np.ascontiguousarray(A[:m_cpu]))
            b_t = Bt
            a_gpu = np.ascontiguousarray(A[m_cpu:])
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                def bg2(a_gpu=a_gpu):
                    cb, shp = g.gemm_async(a_gpu, B, "cs")
                    g.finish_and_get(cb, shp)
                th = threading.Thread(target=bg2)
                th.start()
                _ = a_cpu @ b_t
                th.join()
                ts.append(time.perf_counter() - t0)
            split_res[frac] = min(ts)

        best_split = min(split_res.items(), key=lambda kv: kv[1])
        res[f"n{n}"] = dict(
            cpu_ms=round(cpu_t * 1e3, 2), igpu_ms=round(gpu_t * 1e3, 2),
            seq2_ms=round(seq_t * 1e3, 2), conc2_ms=round(conc_t * 1e3, 2),
            conc_speedup_vs_best=round(min(cpu_t, gpu_t) / conc_t, 2),
            conc_efficiency=round(min(cpu_t, gpu_t) / conc_t / 2, 2),
            split_ms={str(k): round(v * 1e3, 2) for k, v in split_res.items()},
            best_split_frac=best_split[0], best_split_ms=round(best_split[1] * 1e3, 2),
            best_split_speedup_vs_best=round(min(cpu_t, gpu_t) / best_split[1], 2),
        )
        print(f"  n={n}: CPU {cpu_t*1e3:.2f} ms | iGPU {gpu_t*1e3:.2f} ms | "
              f"顺序(两个) {seq_t*1e3:.2f} ms | 并发(两个) {conc_t*1e3:.2f} ms "
              f"(相对最好单引擎 {min(cpu_t,gpu_t)/conc_t:.2f}×)", flush=True)
        print(f"        按行切分并发: " + ", ".join(f"{k:.0%}→{v*1e3:.2f} ms" for k, v in split_res.items())
              + f" 最佳 {best_split[0]:.0%} ({best_split[1]*1e3:.2f} ms, "
                f"相对最好单引擎 {min(cpu_t,gpu_t)/best_split[1]:.2f}×)", flush=True)

    with open(rf"{OUT}\cpu_igpu_concurrency.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
