# -*- coding: utf-8 -*-
"""igpu_cpu_pipeline.py — 实际能用的并发形态：核显算 GEMM 的同时，CPU 解码/预处理音频。

流水线：8 个 chunk，每个 chunk 需要
  CPU 侧：读 wav + 重采样/取帧（内存轻、IO 为主）
  iGPU 侧：一个 2048³ GEMM（代表"模型那部分算力"）
对比三种跑法：
  1) 全顺序（CPU 全做完再 GPU）
  2) 生产者-消费者重叠（CPU 解码线程 + iGPU GEMM 线程，双缓冲）
  3) 纯 GPU（不做 CPU 工作）——上界参考
"""
from __future__ import annotations

import ctypes
import json
import queue
import threading
import time

import numpy as np
import pyopencl as cl
import soundfile as sf

DLL = r"D:\work\clblast_build\CLBlast-master\build\clblast.dll"
OUT = r"D:\work\research\bench_results\audio_lab"
WAVS = [r"D:\work\research\bench_results\audio_lab\long_5min.wav",
        r"D:\work\research\bench_results\audio_lab\piper_en_lessac_long.wav"]
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
        self.bufs = {}

    def gemm(self, a, b):
        a = np.ascontiguousarray(a, np.float32)
        b = np.ascontiguousarray(b, np.float32)
        m, k = a.shape
        _, n = b.shape
        for nm, arr in (("a", a), ("b", b), ("c", np.empty(m * n, np.float32))):
            buf = self.bufs.get(nm)
            nb = arr.nbytes if nm != "c" else m * n * 4
            if buf is None or buf.size < nb:
                buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nb)
                self.bufs[nm] = buf
            if nm != "c":
                cl.enqueue_copy(self.q, buf, arr)
        rc = self.bl.CLBlastSgemm(ROW, NO, NO, m, n, k, ctypes.c_float(1.0),
                                  ctypes.c_void_p(int(self.bufs["a"].int_ptr)), 0, k,
                                  ctypes.c_void_p(int(self.bufs["b"].int_ptr)), 0, n, ctypes.c_float(0.0),
                                  ctypes.c_void_p(int(self.bufs["c"].int_ptr)), 0, n, self.qp, None)
        if rc:
            raise RuntimeError(rc)


def io_work(i):
    """模拟真实 CPU 预处理：读整段 wav + 重采样到 16k + log-mel（IO+CPU 混合）。"""
    import librosa
    p = WAVS[i % len(WAVS)]
    d, sr = sf.read(p, dtype="float32")
    if d.ndim > 1:
        d = d.mean(1)
    d16 = librosa.resample(d, orig_sr=sr, target_sr=16000)
    _ = librosa.feature.melspectrogram(y=d16, sr=16000, n_mels=64)


def main():
    g = G()
    n = 2048
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    g.gemm(A, B); g.q.finish()
    io_work(0)
    chunks = 8

    # 1) 全顺序
    t0 = time.perf_counter()
    for i in range(chunks):
        io_work(i)
    io_total = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(chunks):
        g.gemm(A, B)
    g.q.finish()
    gemm_total = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(chunks):
        io_work(i)
        g.gemm(A, B)
        g.q.finish()
    seq = time.perf_counter() - t0

    # 2) 生产者-消费者重叠（双缓冲）
    q = queue.Queue(maxsize=2)
    stop = threading.Event()

    def producer():
        for i in range(chunks):
            io_work(i)
            q.put(i)
        q.put(None)

    def consumer():
        while True:
            item = q.get()
            if item is None:
                break
            g.gemm(A, B)
            g.q.finish()

    t0 = time.perf_counter()
    tp = threading.Thread(target=producer)
    tc = threading.Thread(target=consumer)
    tp.start(); tc.start(); tp.join(); tc.join()
    pipe = time.perf_counter() - t0

    ideal = max(io_total, gemm_total)
    res = dict(chunks=chunks, io_total_ms=round(io_total * 1e3, 1), gemm_total_ms=round(gemm_total * 1e3, 1),
               sequential_ms=round(seq * 1e3, 1), pipeline_ms=round(pipe * 1e3, 1),
               speedup=round(seq / pipe, 2), ideal_parallel_ms=round(ideal * 1e3, 1),
               efficiency_vs_ideal=round(ideal / pipe, 2))
    print(f"  纯 CPU(IO) {io_total*1e3:6.1f} ms | 纯 iGPU(GEMM) {gemm_total*1e3:6.1f} ms | "
          f"顺序 {seq*1e3:6.1f} ms | 流水线 {pipe*1e3:6.1f} ms → {seq/pipe:.2f}× "
          f"(理想 {ideal*1e3:.1f} ms)", flush=True)
    with open(rf"{OUT}\cpu_igpu_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
