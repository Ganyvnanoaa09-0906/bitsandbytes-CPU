"""Would cache-blocking still buy 1.81x on the shapes the probes actually found?

7.89 measured the effect on one synthetic shape, N=2736 K=512, with a hand-written OpenCL kernel that
reaches 393 GFLOPS solo. Two things make that insufficient as an answer.

First the shape. 7.68 already found that M-chunking wins at K=2048 and loses at K=512, so a result at
one shape does not transfer by assumption. The probes then produced the shapes that run: Qwen3-1.7B's
gate_proj is [6144, 2048] and its down_proj is [2048, 6144], with M=128 for prefill.

Second the kernel. igpu_real_shapes.py established that this iGPU reaches 644.9 GFLOPS under CLBlast
against 393 under the hand-written one, so the earlier share figures understate what the device can
hold. CLBlast is used here.

The design of 7.89 is kept, because it is what made the measurement reproducible: the iGPU's share is a
ratio against a solo baseline taken immediately adjacent, since the solo figure drifts and cross-session
comparisons were what produced a factor-of-2.9 disagreement earlier.

Not run alongside anything else. The first training attempt died of OOM while this kind of probe ran in
parallel, and the failure appeared as a truncated log rather than an error, so the check is arithmetic
first: peak RSS of both jobs must sum to under 80% of what is free.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import ctypes
import os
import statistics
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'

import pyopencl as cl  # noqa: E402
from measure_guard import assert_idle, stamp  # noqa: E402

if not os.path.exists(DLL):
    raise SystemExit('CLBlast 不在 %s —— 而它是本机核显最快的路径(644.9 vs 393 GFLOPS)' % DLL)

bl = ctypes.CDLL(DLL)
bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
]
LAYOUT_ROW, TRANSPOSE_NO = 1, 111

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(queue.int_ptr))
torch.set_num_threads(5)

# ★ 形状来自 §7.90/§7.93 两个探针的实测观测,不是造的
SHAPES = [
    ('gate_proj [6144,2048] M=128', 128, 6144, 2048),
    ('up_proj   [6144,2048] M=128', 128, 6144, 2048),
    ('down_proj [2048,6144] M=128', 128, 2048, 6144),
    ('q_proj    [2048,2048] M=128', 128, 2048, 2048),
    ('gate_proj [6144,2048] M=1', 1, 6144, 2048),
]

print()
print('  设备 %s · %d CU · %d MB' % (dev.name, dev.max_compute_units,
                                     dev.global_mem_size // 2 ** 20))
stamp('真实形状上,分块并发还成不成立(CLBlast)')
print()
print('  %-30s %8s %10s %10s %10s %10s %9s' %
      ('形状', '工作集', 'CPU单独', '单独基准', '并发CPU', '并发核显', '核显占比'))
print('  ' + '-' * 96)

rows = []
for name, M, N, K in SHAPES:
    fl = 2.0 * M * N * K
    rng = np.random.default_rng(5)
    Anp = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
    Bnp = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Anp)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Bnp)
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    sargs = (LAYOUT_ROW, TRANSPOSE_NO, TRANSPOSE_NO, M, N, K, ctypes.c_float(1.0),
             ctypes.c_void_p(int(ba.int_ptr)), 0, K,
             ctypes.c_void_p(int(bb.int_ptr)), 0, N,
             ctypes.c_float(0.0),
             ctypes.c_void_p(int(bc.int_ptr)), 0, N,
             ctypes.byref(qp), None)

    At = torch.from_numpy(Anp)
    Bt = torch.from_numpy(Bnp)
    Ct = torch.empty(M, N)

    def cpu_full():
        torch.mm(At, Bt, out=Ct)

    tm = tn = 256

    def cpu_tiled():
        for i in range(0, M, tm):
            ie = min(i + tm, M)
            for j in range(0, N, tn):
                je = min(j + tn, N)
                torch.mm(At[i:ie], Bt[:, j:je], out=Ct[i:ie, j:je])

    def gpu_once():
        """★ 必须检查返回码。第一版漏了它,于是 CLBlast 报了错就立刻返回,
        而"耗时"只剩下入队开销 ⇒ 算出 46787 GFLOPS,是这台设备峰值(1150)的 40 倍。
        ★ 而这个错我在本项目的核显早期【已经犯过一次】(33446 GFLOPS 那次),
        记录就写在报告里 —— 而我还是又犯了一遍。判据应当是:
        【任何超过设备标称峰值的核显数字,先当测量错误处理,而不是当发现。】"""
        rc = bl.CLBlastSgemm(*sargs)
        if rc != 0:
            raise RuntimeError('CLBlast 返回 %d（%s）—— 内核没跑,而"耗时"只是入队'
                               % (rc, 'CL_OUT_OF_RESOURCES' if rc == -1016 else '见 CLBlast 错误表'))
        queue.finish()

    def gpu_solo(reps=6):
        gpu_once()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            gpu_once()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    for _ in range(2):
        cpu_full()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        cpu_full()
        ts.append(time.perf_counter() - t0)
    cpu_f = fl / statistics.median(ts) / 1e9

    for _ in range(2):
        cpu_tiled()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        cpu_tiled()
        ts.append(time.perf_counter() - t0)
    cpu_t = fl / statistics.median(ts) / 1e9
    ws = (tm * K + K * tn + tm * tn) * 4 / 1024 ** 2

    s_b = fl / gpu_solo() / 1e9
    stop = [False]
    cnt = [0]

    def worker():
        while not stop[0]:
            cpu_tiled()
            cnt[0] += 1

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(2.0)
    cnt[0] = 0
    t0 = time.perf_counter()
    gts = []
    for _ in range(6):
        tg = time.perf_counter()
        rc = bl.CLBlastSgemm(*sargs)
        if rc != 0:
            raise RuntimeError('并发测量里 CLBlast 返回 %d' % rc)
        queue.finish()
        gts.append(time.perf_counter() - tg)
    wall = time.perf_counter() - t0
    stop[0] = True
    th.join(timeout=5)
    time.sleep(2.0)
    s_a = fl / gpu_solo() / 1e9
    s_ref = statistics.mean([s_b, s_a])
    cpu_c = cnt[0] * fl / wall / 1e9
    gpu_c = fl / statistics.median(gts) / 1e9
    rows.append((name, ws, cpu_f, cpu_t, s_ref, cpu_c, gpu_c))

    print('  %-30s %6.1f MB %10.1f %10.1f %10.1f %10.1f %8.0f%%'
          % (name, ws, cpu_t, s_ref, cpu_c, gpu_c, gpu_c / s_ref * 100))

print()
print('  %-30s %12s %12s %12s %12s' % ('形状', '合计并发', 'CPU 单独', '单独基准', '合计/CPU单独'))
print('  ' + '-' * 84)
for r in rows:
    tot = r[5] + r[6]
    print('  %-30s %12.1f %12.1f %12.1f %11.2f×' % (r[0], tot, r[3], r[4], tot / r[3]))
print()
best = max(rows, key=lambda r: (r[5] + r[6]) / r[3])
tot = best[5] + best[6]
print('  最好: %s ⇒ 合计 %.1f 对 CPU 单独 %.1f ⇒ 【%.2f×】' % (best[0], tot, best[3], tot / best[3]))
print('  而 §7.89 在合成形状(N=2736,K=512,手写内核)上得 【1.81×】')
print()
print('  ⇒ 判据: 合计 > CPU 单独 × 1.20 ⇒ 值得做')
gap = [r for r in rows if r[2] > r[3] * 1.05]
if gap:
    print('  ★ 注意这些形状上【分块让 CPU 自己变慢了】:')
    for r in gap:
        print('    %-30s 全量 %.1f → 分块 %.1f (%.2f×)' % (r[0], r[2], r[3], r[3] / r[2]))
print()
