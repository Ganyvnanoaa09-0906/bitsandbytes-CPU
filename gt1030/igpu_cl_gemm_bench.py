# -*- coding: utf-8 -*-
"""igpu_cl_gemm_bench.py — OpenCL/CLBlast 的 SGEMM 实测（对比 CPU torch matmul）。

CLBlast 只有 GEMM；卷积要先 im2col 再 GEMM。这里先量裸 GEMM 的 GFLOPS，
判断核显在这台机器上到底有多少算力可用。
枚举值来自仓库里已验证的封装：ROW=101, NO=111, YES=112。
"""
from __future__ import annotations

import ctypes
import json
import time

import numpy as np
import pyopencl as cl
import torch

DLL = r"D:\work\clblast_build\CLBlast-master\build\clblast.dll"
OUT = r"D:\work\research\bench_results\audio_lab"
ROW, NO, YES = 101, 111, 112

torch.set_num_threads(6)


def setup():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    bl = ctypes.CDLL(DLL)
    bl.CLBlastSgemm.restype = ctypes.c_int
    bl.CLBlastSgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    qp = ctypes.c_void_p(int(queue.int_ptr))
    return ctx, queue, bl, qp


def main():
    ctx, queue, bl, QP = setup()
    dev = ctx.devices[0]
    print(f"  设备 {dev.name} | CU {dev.max_compute_units} | CLBlast ok", flush=True)

    rows = {}
    for n in (256, 512, 1024, 2048):
        a_np = np.random.rand(n, n).astype(np.float32)
        b_np = np.random.rand(n, n).astype(np.float32)
        c_np = np.empty((n, n), np.float32)
        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_np)
        c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, c_np.nbytes)

        def run():
            bl.CLBlastSgemm(ROW, NO, NO, n, n, n, ctypes.c_float(1.0),
                            ctypes.c_void_p(int(a_buf.int_ptr)), 0, n,
                            ctypes.c_void_p(int(b_buf.int_ptr)), 0, n,
                            ctypes.c_float(0.0),
                            ctypes.c_void_p(int(c_buf.int_ptr)), 0, n,
                            QP, None)

        run()
        queue.finish()
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            run()
        queue.finish()
        dt = (time.perf_counter() - t0) / reps
        gflops = 2 * n ** 3 / dt / 1e9

        ta, tb = torch.from_numpy(a_np), torch.from_numpy(b_np)
        tc = ta @ tb
        t1 = time.perf_counter()
        for _ in range(reps):
            tc = ta @ tb
        dtt = (time.perf_counter() - t1) / reps
        cpu_gflops = 2 * n ** 3 / dtt / 1e9

        # 正确性
        cl.enqueue_copy(queue, c_np, c_buf).wait()
        err = float(np.abs(c_np - tc.numpy()).max())
        rows[n] = dict(igpu_ms=round(dt * 1e3, 2), igpu_gflops=round(gflops, 1),
                       cpu_ms=round(dtt * 1e3, 2), cpu_gflops=round(cpu_gflops, 1),
                       max_err=round(err, 5))
        print(f"  n={n:5d}  iGPU {dt*1e3:8.2f} ms ({gflops:6.1f} GFLOPS) | "
              f"CPU {dtt*1e3:8.2f} ms ({cpu_gflops:6.1f} GFLOPS) | err {err:.4f}", flush=True)

    with open(rf"{OUT}\cl_gemm_bench.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("RESULT " + json.dumps(rows))


if __name__ == "__main__":
    main()
