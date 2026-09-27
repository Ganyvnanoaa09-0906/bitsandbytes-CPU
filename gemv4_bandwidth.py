# -*- coding: utf-8 -*-
"""gemv4_bandwidth.py — 4bit GEMV 到底卡在哪：内存带宽还是译码运算？

判据（criterion）:
    4bit GEMV 每搬 1 字节权重能达到的吞吐，与【同一台机器、同一时段】实测的
    纯内存拷贝带宽相比。接近则说明它已吃满带宽，进一步压位数(2bit/3bit)才有意义。

oracle:
    · 带宽上限现场实测（memcpy），不引用任何外部常数 —— 避免跨口径比较。
    · 同时测稠密 fp32 GEMV 作为交叉校验：它必然带宽受限，若它也不达标，
      说明是【测量方法】有问题，而不是被测量的内核有问题。

⚠️ 已知口径缺陷（实测暴露，保留在代码里当警示）:
    把 DRAM 上限当作"天花板"去衡量一个能从 L3 取数的内核，会算出 >100% 的荒谬值。
    实测：64MB 拷贝 15.54 GB/s，而 M=1 GEMV 达 19.67 GB/s（19MB 权重部分命中 L3）。
    ⇒ %ceiling 列**只在权重远大于 L3 时有意义**；权重小于 L3 时看绝对 GB/s。

结论（R5-4500U，6 线程，2026-09-27）:
    M=1  0.960 ms / 19.67 GB/s   —— 带宽导向，压位数有空间
    M=8  6.076 ms /  3.13 GB/s
    M=64 37.919 ms / 0.53 GB/s   —— 只有稠密 fp32 的 10.6%
    ⇒ 该核是为解码(M=1)设计的；训练/大 batch 下 4bit 权重不划算，该用 fp16。

data_type 约定（实测，report 10.147）: 1 -> FP4 表, 2 -> NF4 表。
"""
from __future__ import annotations

import ctypes as ct
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _pkg_parent():
    """返回包的【父】目录，即 <repo>\\bitsandbytes（其下才是 bitsandbytes\\__init__.py）。

    不能把 <repo> 放进 sys.path：仓库根下也有一个叫 bitsandbytes 的目录，会被当成
    命名空间包（__file__ = None）抢先匹配，`bitsandbytes.cextension` 就找不到了。
    fused_cpu.py:35 记的是同一个坑。
    """
    bases = [HERE]
    b = HERE
    for _ in range(3):
        b = os.path.dirname(b)
        bases.append(b)
    for base in bases:
        if not base:
            continue
        parent = os.path.join(base, "bitsandbytes")
        pkg = os.path.join(parent, "bitsandbytes")
        if os.path.isfile(os.path.join(pkg, "__init__.py")) and \
           os.path.isfile(os.path.join(pkg, "cextension.py")):
            return parent
    return None


_PKG = _pkg_parent()
if _PKG is None:
    raise SystemExit(
        "cannot locate the bitsandbytes package (needs __init__.py + cextension.py); "
        "looked near: " + HERE
    )

# _pkg_parent() 返回的【就是】要放进 sys.path 的那个目录（包的父目录）——不要再
# dirname 一次，否则会退回到仓库根，而仓库根下同名的 bitsandbytes 目录会被当成
# 命名空间包抢先匹配（__file__ = None），`bitsandbytes.cextension` 就找不到了。
# 这正是上一版失败的原因：名字取对了，路径多剥了一层。
_PKG_PARENT = _PKG

# 把仓库根（= 脚本所在目录，Python 会把它放在 sys.path[0]）摘掉，再插入包的父目录。
_here_abs = os.path.abspath(HERE)
_kept = []
for p in sys.path:
    try:
        if p and os.path.abspath(p) == _here_abs:
            continue
    except OSError:
        pass
    _kept.append(p)
sys.path[:] = _kept
sys.path.insert(0, _PKG_PARENT)

from bitsandbytes.cextension import lib  # noqa: E402
from bitsandbytes.functional import get_ptr  # noqa: E402

assert lib is not None, "libbitsandbytes_cpu not loaded"

lib.cgemv_4bit_inference_cpu_fp32.argtypes = [
    ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
    ct.c_longlong, ct.c_longlong, ct.c_longlong,
    ct.c_longlong, ct.c_longlong, ct.c_longlong, ct.c_longlong, ct.c_int,
]
lib.cgemv_4bit_inference_cpu_fp32.restype = None


def timed(fn, iters):
    """返回 (median_ms, min_ms)。

    ⚠️ 跨进程比时间在本项目已制造过两次假结论；本脚本所有数字都取自同一进程。
    """
    fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def memcpy_bw(nbytes, iters=25):
    src = torch.empty(nbytes, dtype=torch.uint8)
    dst = torch.empty(nbytes, dtype=torch.uint8)
    dst.copy_(src)
    _, best = timed(lambda: dst.copy_(src), iters)
    return 2.0 * nbytes / (best * 1e-3) / 1e9, best


def gemv4_case(M, N, K, data_type, blocksize, iters=12):
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    absmax = torch.rand(N, (K + blocksize - 1) // blocksize, dtype=torch.float32) + 0.5
    out = torch.empty(M, N, dtype=torch.float32)
    total = B.numel() + absmax.numel() * 4 + out.numel() * 4

    def fn():
        lib.cgemv_4bit_inference_cpu_fp32(
            get_ptr(A), get_ptr(B), get_ptr(absmax), get_ptr(out),
            M, N, K, A.stride(0), K // 2, out.stride(0), blocksize, data_type,
        )

    _, best = timed(fn, iters)
    return best, total / (best * 1e-3) / 1e9, total


def dense_case(M, N, K, iters=12):
    A = torch.randn(M, K)
    W = torch.randn(N, K)
    _, best = timed(lambda: A @ W.t(), iters)
    total = W.numel() * 4 + A.numel() * 4 + M * N * 4
    return best, total / (best * 1e-3) / 1e9, total


def main():
    print("=" * 78)
    print("4bit GEMV: memory bandwidth vs decode cost -- feasibility criterion")
    print("=" * 78)
    print("threads(torch) = %d" % torch.get_num_threads())
    print("data_type: 1 -> FP4 table, 2 -> NF4 table (measured; report 10.147)")

    print("\n[oracle] pure memcpy bandwidth (read+write, min of N):", flush=True)
    ceilings = {}
    for mb in (4, 16, 64):
        bw, best = memcpy_bw(mb << 20)
        ceilings[mb] = bw
        print("  %4d MB: %8.2f GB/s   (best %.3f ms)" % (mb, bw, best), flush=True)
    ref_bw = ceilings[16]
    print("  => using the 16 MB figure %.2f GB/s as reference "
          "(an L3-resident kernel can legitimately exceed it)" % ref_bw, flush=True)

    print("\n[cross-check] dense fp32 GEMV (must be bandwidth bound):", flush=True)
    for (M, N, K) in ((1, 4096, 4096), (8, 4096, 4096)):
        best, bw, _ = dense_case(M, N, K)
        print("  M=%3d N=%d K=%d: %8.3f ms  %7.2f GB/s  = %5.1f%% of ceiling"
              % (M, N, K, best, bw, 100 * bw / ref_bw), flush=True)

    print("\n[4bit GEMV]:", flush=True)
    print("  %4s %4s %6s %7s %7s %9s %8s %7s %7s"
          % ("type", "M", "N", "K", "MB", "ms", "GB/s", "%ceil", "%dense"), flush=True)
    for data_type, name in ((1, "FP4"), (2, "NF4")):
        for (N, K) in ((4096, 8192), (4096, 16384)):
            for M in (1, 8, 64):
                best, bw, total = gemv4_case(M, N, K, data_type, 64)
                _, dbw, _ = dense_case(M, N, K)
                print("  %4s %4d %6d %7d %7.1f %9.3f %8.2f %6.1f%% %6.1f%%"
                      % (name, M, N, K, total / 1e6, best, bw,
                         100 * bw / ref_bw, 100 * bw / dbw), flush=True)
            print(flush=True)

    print("=" * 78)
    print("Reading: %ceil is only meaningful when weights >> L3. If time grows")
    print("linearly with M at constant traffic, the kernel is compute bound and")
    print("more aggressive quantization will not help -- use fp16 weights.")
    print("=" * 78)


if __name__ == "__main__":
    main()
