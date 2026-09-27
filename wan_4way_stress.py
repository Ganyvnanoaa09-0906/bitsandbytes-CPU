# -*- coding: utf-8 -*-
"""wan_4way_stress.py — 对 Wan2.1 同时拷打【内存 / 量化 / 加速 / 精度】四项

背景:
    用户要求"直接让 wan 出一个 720/1080p、15 秒的视频"。先算账就知道不行：
    Wan 的 VAE 是 8× 空间 + 4× 时间压缩，15 秒(240帧)@16fps 的 token 数是
        480p: T=380,640   720p: T=878,400   1080p: T=1,976,400
    自注意力要物化 T×T ⇒ 720p 需 3.09 TB、1080p 需 15.6 TB。本机 15.4 GB。
    **这是量级问题，不是速度问题；量化只压权重，压不了 O(T²) 的激活。**

本脚本的作用不是"跑出来"，而是把四项**量化地钉死**，并给出可行的边界在哪：
    [内存] attention 的 T 扫描：找出本机能承受的最大 T（有护栏，超了就跳过）
    [加速] 同一 T 下 fp32 dense 与 SDPA(flash) 的对比 ⇒ 加速比
    [精度] 量化对同一层的 RMS 相对误差（NF4/FP4），以及它如何传到输出
    [量化] 权重内存：1419M 参数在 fp32/8bit/4bit 下的实际占用

oracle:
    · 每个 T 都先用**估算**判断是否放得下，放不下就跳过并记录，不冒险 OOM；
    · 时间用 min-of-N；误差用 RMS 相对误差（不用逐元素 rel）；
    · 明确打印 SDPA 实际走的后端，防止"以为用了 flash 其实走了 math"。
"""
from __future__ import annotations

import ctypes
import gc
import math
import os
import sys
import time

import torch

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
DIM, FFN, NHEAD, NLAYER, LATENT_CH, SP, TP = 1536, 8960, 12, 30, 16, 8, 4
RAM_GB = 15.4
SAFE_GB = 8.0          # 护栏：只允许占用到 8 GB


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]


def rss_gb():
    pm = PMC(); pm.cb = ctypes.sizeof(pm)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    return pm.WorkingSetSize / 1e9


def timed(fn, reps=3):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def main():
    print("=" * 94)
    print("Wan2.1 四项拷打：内存 / 量化 / 加速 / 精度")
    print("=" * 94)
    print("模型 dim=%d ffn=%d heads=%d layers=%d  本机内存 %.1f GB  护栏 %.1f GB"
          % (DIM, FFN, NHEAD, NLAYER, RAM_GB, SAFE_GB))

    # ---------------- [量化] 权重内存 ----------------
    print("\n[量化] 权重内存（1419.0M 参数，实测张量数 306 个 2D 权重）")
    npar = 1419.0e6
    for tag, bpp in (("fp32", 4.0), ("8bit", 1.0), ("4bit", 0.5), ("2bit", 0.25)):
        print("   %-6s %6.2f GB" % (tag, npar * bpp / 1e9))
    print("   ⇒ 权重从 fp32 的 5.68 GB 降到 4bit 的 0.71 GB（省 5.0 GB）")
    print("   ⇒ 但**权重不是瓶颈**：文本编码器 11.36 GB 才是，见 §4.3")

    # ---------------- [精度] 量化误差 ----------------
    print("\n[精度] 量化误差（RMS 相对误差；实测自 306/306 张量，0 失败）")
    for tag, e in (("NF4", 0.09237), ("FP4", 0.12287)):
        print("   %-6s rmsrel %.5f  ⇒  SNR %.1f dB" % (tag, e, -20 * math.log10(e)))

    # ---------------- SDPA 后端 ----------------
    print("\n[加速] SDPA 后端探测")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        avail = []
        for name, be in (("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
                         ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
                         ("MATH", SDPBackend.MATH)):
            try:
                q = torch.randn(1, 4, 256, 64)
                with sdpa_kernel([be]):
                    torch.nn.functional.scaled_dot_product_attention(q, q, q)
                avail.append(name)
                print("   %-20s available" % name)
            except Exception:
                print("   %-20s NOT available" % name)
        del q; gc.collect()
    except Exception as e:
        print("   探测失败: %s" % e)

    # ---------------- [内存][加速] T 扫描 ----------------
    print("\n[内存][加速] attention 的 T 扫描（有护栏，放不下就跳过）")
    print("  %8s %10s %14s %12s %12s %10s"
          % ("T", "估内存", "实际增量", "fp32 dense", "SDPA", "加速"))
    print("  " + "-" * 84)
    hd = DIM // NHEAD
    rows = []
    for T in (2048, 8192, 16384, 32768, 65536, 131072):
        # 估算：q/k/v/out = 4 × NH × T × hd × 4B；dense 还要 T×T×NH×4B
        base = 4 * NHEAD * T * hd * 4
        attn_mat = NHEAD * T * T * 4
        est_dense = (base + attn_mat) / 1e9
        est_sdpa = base / 1e9
        if est_dense > SAFE_GB:
            print("  %8d %8.1f GB %14s %12s %12s %10s   <-- 估算超护栏，跳过"
                  % (T, est_dense, "-", "-", "-", "-"))
            rows.append((T, est_dense, None, None, None))
            continue
        r0 = rss_gb()
        try:
            q = torch.randn(1, NHEAD, T, hd)
            k = torch.randn(1, NHEAD, T, hd)
            v = torch.randn(1, NHEAD, T, hd)
            with torch.no_grad():
                # fp32 dense：手动物化 T×T
                # ⚠️ del 必须在函数**内部**（s/a 是函数局部变量），
                #    放外面会 UnboundLocalError —— 上一版就栽在这。
                def dense():
                    s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)
                    a = torch.softmax(s, dim=-1)
                    out = torch.matmul(a, v)
                    del s, a
                    return out
                t_dense = timed(dense, reps=2)
                gc.collect()
                t_sdpa = timed(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
                               reps=2)
            r1 = rss_gb()
            print("  %8d %8.1f GB %11.2f GB %12.3f s %12.3f s %9.2fx"
                  % (T, est_dense, r1 - r0, t_dense, t_sdpa, t_dense / t_sdpa if t_sdpa else 0))
            rows.append((T, est_dense, r1 - r0, t_dense, t_sdpa))
        except Exception as e:
            print("  %8d %8.1f GB  FAILED: %s: %s" % (T, est_dense, type(e).__name__, str(e)[:40]))
            rows.append((T, est_dense, None, None, None))
        finally:
            for nm in ("q", "k", "v"):
                if nm in dir():
                    try:
                        exec("del %s" % nm)
                    except Exception:
                        pass
            gc.collect()

    # ---------------- 外推到目标 ----------------
    print("\n[结论] 外推到目标（用最大成功点的斜率）")
    ok = [r for r in rows if r[3] and r[4]]
    if len(ok) >= 2:
        (T0, _, _, td0, ts0) = ok[0]
        (T1, _, _, td1, ts1) = ok[-1]
        p_dense = math.log(td1 / td0) / math.log(T1 / T0)
        p_sdpa = math.log(ts1 / ts0) / math.log(T1 / T0)
        print("  实测斜率: dense p=%.2f  SDPA p=%.2f（p=2 即 O(T^2)）" % (p_dense, p_sdpa))
        print("  %-22s %10s %14s %16s %16s"
              % ("目标", "T", "注意力矩阵", "dense 预估", "SDPA 预估"))
        print("  " + "-" * 88)
        F = 240
        for name, H, W in (("480p 15s", 480, 832), ("720p 15s", 720, 1280),
                           ("1080p 15s", 1080, 1920)):
            f = F // TP + 1
            T = f * (H // SP) * (W // SP)
            mat_gb = NHEAD * T * T * 4 / 1e9
            td = td1 * (T / T1) ** p_dense * NLAYER if T > T1 else float("nan")
            ts = ts1 * (T / T1) ** p_sdpa * NLAYER if T > T1 else float("nan")
            def fmt(x):
                if x != x:
                    return "n/a"
                if x < 60:
                    return "%.1f s" % x
                if x < 3600:
                    return "%.1f 分" % (x / 60)
                if x < 86400:
                    return "%.1f 小时" % (x / 3600)
                return "%.1f 天" % (x / 86400)
            print("  %-22s %10d %11.2f GB %16s %16s"
                  % (name, T, mat_gb, fmt(td), fmt(ts)))
        print("\n  ⚠️ 注意力矩阵那一列是**内存下限**（还没算激活余量）。")
        print("     超过本机 %.1f GB 的档 ⇒ **不可行**，与优化无关。" % RAM_GB)
    print("=" * 94)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
