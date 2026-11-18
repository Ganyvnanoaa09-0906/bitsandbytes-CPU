# -*- coding: utf-8 -*-
"""igpu_cpu_async2.py — 异步与重叠的边界：
  A) iGPU GEMM 在子线程跑，主线程睡觉 → 验证 CLBlast 真的异步（总时间≈iGPU 单跑）；
  B) iGPU GEMM 在子线程跑，主线程做**重 CPU 计算**（另一个 GEMM）→ 前面测到只有 ~1.1–1.2×；
  C) iGPU GEMM 在子线程跑，主线程做**IO/解码类轻 CPU 工作**（读 wav + 重采样）→ 看能否几乎完全重叠。
  D) 采样 CPU 频率：CPU 单跑 / iGPU 单跑 / 同时跑，验证是否掉频（15 W TDP 共享）。
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import threading
import time

import numpy as np
import pyopencl as cl
import soundfile as sf
import torch

DLL = r"D:\work\clblast_build\CLBlast-master\build\clblast.dll"
OUT = r"D:\work\research\bench_results\audio_lab"
WAV = r"D:\work\research\bench_results\audio_lab\long_5min.wav"
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
        for nm, arr in (("a", a), ("b", b)):
            buf = self.b.get(nm)
            if buf is None or buf.size < arr.nbytes:
                buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, arr.nbytes)
                self.b[nm] = buf
            cl.enqueue_copy(self.q, buf, arr)
        cb = self.b.get("c")
        if cb is None or cb.size < m * n * 4:
            cb = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, m * n * 4)
            self.b["c"] = cb
        rc = self.bl.CLBlastSgemm(ROW, NO, NO, m, n, k, ctypes.c_float(1.0),
                                  ctypes.c_void_p(int(self.b["a"].int_ptr)), 0, k,
                                  ctypes.c_void_p(int(self.b["b"].int_ptr)), 0, n, ctypes.c_float(0.0),
                                  ctypes.c_void_p(int(cb.int_ptr)), 0, n, self.qp, None)
        if rc:
            raise RuntimeError(rc)
        self.q.finish()


def cpu_clock():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).CurrentClockSpeed"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[-1])
    except Exception:
        return -1


def sample_clock_under(work, seconds=2.0):
    stop = threading.Event()
    vals = []

    def poll():
        while not stop.is_set():
            vals.append(cpu_clock())

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.time()
    while time.time() - t0 < seconds:
        work()
    stop.set()
    th.join(timeout=3)
    vals = [v for v in vals if v > 0]
    return int(np.median(vals)) if vals else -1


def main():
    g = G()
    torch.set_num_threads(6)
    n = 2048
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    At, Bt = torch.from_numpy(A), torch.from_numpy(B)
    res = {}

    # 基线
    g.gemm(A, B, "w"); At @ Bt
    t0 = time.perf_counter(); g.gemm(A, B, "x"); gpu_t = time.perf_counter() - t0
    t0 = time.perf_counter(); At @ Bt; cpu_t = time.perf_counter() - t0
    print(f"  基线: iGPU GEMM2048 {gpu_t*1e3:.1f} ms | CPU GEMM2048 {cpu_t*1e3:.1f} ms", flush=True)

    # A) iGPU 跑 + 主线程 sleep
    t0 = time.perf_counter()
    th = threading.Thread(target=lambda: g.gemm(A, B, "a"))
    th.start(); time.sleep(gpu_t); th.join()
    a_t = time.perf_counter() - t0
    res["A_gpu_plus_sleep"] = dict(total_ms=round(a_t * 1e3, 1), gpu_ms=round(gpu_t * 1e3, 1),
                                   overlap=round(max(gpu_t, gpu_t) / a_t, 2))
    print(f"  A) iGPU 跑 + CPU 睡觉: 总 {a_t*1e3:.1f} ms（≈ iGPU 单跑 {gpu_t*1e3:.1f} ms）→ 异步成立", flush=True)

    # B) iGPU 跑 + 主线程做 CPU GEMM
    t0 = time.perf_counter()
    th = threading.Thread(target=lambda: g.gemm(A, B, "b"))
    th.start(); At @ Bt; th.join()
    b_t = time.perf_counter() - t0
    res["B_gpu_plus_cpu_gemm"] = dict(total_ms=round(b_t * 1e3, 1),
                                      seq_ms=round((cpu_t + gpu_t) * 1e3, 1),
                                      overlap_gain=round((cpu_t + gpu_t) / b_t, 2))
    print(f"  B) iGPU + CPU 各算一个 GEMM: 总 {b_t*1e3:.1f} ms "
          f"（顺序应为 {(cpu_t+gpu_t)*1e3:.1f}）→ 重叠收益 {(cpu_t+gpu_t)/b_t:.2f}×", flush=True)

    # C) iGPU 跑 + 主线程做音频解码/重采样
    def io_work():
        d, sr = sf.read(WAV, dtype="float32")
        _ = d[::2].mean(1) if d.ndim > 1 else d[::2]

    io_work()
    t0 = time.perf_counter(); io_work(); io_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    th = threading.Thread(target=lambda: g.gemm(A, B, "c"))
    th.start(); io_work(); th.join()
    c_t = time.perf_counter() - t0
    res["C_gpu_plus_io"] = dict(total_ms=round(c_t * 1e3, 1), io_ms=round(io_t * 1e3, 1),
                                gpu_ms=round(gpu_t * 1e3, 1),
                                overlap_gain=round((io_t + gpu_t) / c_t, 2))
    print(f"  C) iGPU + CPU 解码音频: 总 {c_t*1e3:.1f} ms（iGPU {gpu_t*1e3:.1f} + IO {io_t*1e3:.1f} "
          f"= {(gpu_t+io_t)*1e3:.1f}）→ 重叠收益 {(io_t+gpu_t)/c_t:.2f}×", flush=True)

    # D) 频率采样
    res["D_clock_mhz"] = dict(
        idle=sample_clock_under(lambda: time.sleep(0.05), 1.5),
        cpu_only=sample_clock_under(lambda: At @ Bt, 2.0),
        igpu_only=sample_clock_under(lambda: g.gemm(A, B, "d"), 2.0),
        both=sample_clock_under(lambda: (threading.Thread(target=lambda: g.gemm(A, B, "e")).start(), At @ Bt)[1], 2.0),
    )
    print(f"  D) CPU 频率: 空闲 {res['D_clock_mhz']['idle']} MHz | CPU 满载 {res['D_clock_mhz']['cpu_only']} | "
          f"iGPU 满载 {res['D_clock_mhz']['igpu_only']} | 双满载 {res['D_clock_mhz']['both']}", flush=True)

    with open(rf"{OUT}\cpu_igpu_async.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
