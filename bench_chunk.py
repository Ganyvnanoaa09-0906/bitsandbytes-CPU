"""Chunk-size sensitivity: speed & peak memory vs GDN_CPU_CHUNK (C).

Usage: PYTHONPATH=/workspace/bitsandbytes python bench_chunk.py
"""
import os
import time

import torch

B, T, H, K, V = 1, 8192, 16, 128, 128

torch.manual_seed(0)
q = torch.randn(B, T, H, K)
k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
v = torch.randn(B, T, H, V) * 0.5
beta = torch.rand(B, T, H) * 0.9 + 0.05
g = -torch.rand(B, T, H) * 0.1
for t in (q, k, v):
    t.requires_grad_(True)

from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule  # noqa: E402


def rss_mb():
    if os.name == "nt":  # Windows: working set via psapi (no /proc)
        try:
            import ctypes
            import ctypes.wintypes as wt

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t)] + \
                           [(n, ctypes.c_size_t) for n in (
                               "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                               "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                               "PagefileUsage", "PeakPagefileUsage")]

            pmc = _PMC(cb=ctypes.sizeof(_PMC))
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return float(pmc.WorkingSetSize) / (1 << 20)
        except Exception:
            pass
        return -1.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return -1


print(f"{'C':>5} {'ckpt MB':>8} {'fwd ms':>8} {'bwd ms':>8} {'peak dRSS MB':>13}")
for C in (32, 64, 128, 148, 256, 512):
    os.environ["GDN_CPU_CHUNK"] = str(C)
    base = rss_mb()
    q.grad = k.grad = v.grad = None
    t0 = time.perf_counter()
    o = fused_recurrent_gated_delta_rule(q, k, v, beta, g)[0]
    t1 = time.perf_counter()
    o.square().mean().backward()
    t2 = time.perf_counter()
    peak = rss_mb() - base
    del o
    ckpt = B * H * ((T + C - 1) // C) * K * V * 4 / 1e6
    print(f"{C:>5} {ckpt:>8.1f} {(t1-t0)*1e3:>8.1f} {(t2-t1)*1e3:>8.1f} {peak:>13.1f}")
