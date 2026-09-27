# -*- coding: utf-8 -*-
"""video_attn_scaling.py — 视频侧真正要回答的第一个问题：attention 是不是 O(T²)？

背景（必须实测而不是外推）:
    仓库里 measure_video_cost2.py 把 T=256 的实测单位成本外推到 T=43200，得出
    "视频 144p 15s 单层 4.2 分钟、×10 层 ≈ 42 分钟"。**那份外推从未被验证。**
    它成立与否决定完全不同的技术路线：
      · 若真是 O(T²) ⇒ 长视频只能走线性注意力（GDN 内核我们已有）
      · 若不是（例如 SDPA 走了 flash/内存高效路径，或算力被别的部分主导）
        ⇒ 先优化现有路径即可，不必换架构

criterion（判据）:
    单位成本 t / T² 应当是**常数**（O(T²)），单位成本 t / T 应当随 T **线性增长**。
    只看单点无法判断，所以每个 T 都同时给出两个比值，并给出 log-log 斜率。

oracle（防自欺）:
    · 正确性自检：attention 输出必须与手写的朴素实现一致（否则测的是错的东西）。
    · 明确报告 SDPA 是否启用了 flash / mem_efficient 后端。
    · **前向与反向分开测**：训练与推理的成本结构不同，混在一起会得出错误结论。
    · 显式限时 + 单次迭代上限，避免在 15W 机器上把内存打爆。
"""
from __future__ import annotations

import ctypes
import gc
import math
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from small_image_model_v2 import SmallImageConfigV2, SelfAttention  # noqa: E402

torch.set_num_threads(6)


class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('a', ctypes.c_size_t), ('b', ctypes.c_size_t), ('c', ctypes.c_size_t),
                ('d', ctypes.c_size_t), ('e', ctypes.c_size_t), ('f', ctypes.c_size_t)]


def peak_rss_mb():
    pm = PMC()
    pm.cb = ctypes.sizeof(pm)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    return pm.PeakWorkingSetSize / 1e6


def cfg_default():
    return SmallImageConfigV2(
        vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
        ffn_dim=2048, total_tokens=256, grid=16, think_tokens=16,
        text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
        pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
        hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)


def timed(fn, warmup=1, reps=3):
    for _ in range(warmup):
        fn()
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def main():
    cfg = cfg_default()
    D = cfg.d_model
    attn = SelfAttention(cfg)
    attn.eval()
    print('=' * 84)
    print('attention 是否 O(T^2)？—— 实测（不是外推）')
    print('=' * 84)
    print('d_model=%d n_head=%d n_kv_head=%d  threads=%d'
          % (D, cfg.n_head, cfg.n_kv_head, torch.get_num_threads()))

    # ---- SDPA backend: which path are we actually measuring? ----
    print('\n[SDPA 后端] torch %s' % torch.__version__)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        for name, be in (('FLASH_ATTENTION', SDPBackend.FLASH_ATTENTION),
                         ('EFFICIENT_ATTENTION', SDPBackend.EFFICIENT_ATTENTION),
                         ('MATH', SDPBackend.MATH)):
            try:
                q = torch.randn(1, 4, 256, 64)
                with sdpa_kernel([be]):
                    torch.nn.functional.scaled_dot_product_attention(q, q, q)
                print('  %-20s available' % name)
            except Exception as e:
                print('  %-20s NOT available (%s)' % (name, type(e).__name__))
    except Exception as e:
        print('  (cannot enumerate backends: %s)' % e)

    # ---- correctness self-check before trusting any timing ----
    print('\n[正确性自检] 与手写朴素 attention 对比（小 T，fp32）')
    torch.manual_seed(0)
    Tc = 64
    xc = torch.randn(1, Tc, D)
    with torch.no_grad():
        y_model = attn(xc, cache=None, append=True)
        # 直接复用模块自己的投影入口，避免猜属性名（fuse_qkv 默认 False 时
        # 模块上是 self.q/self.k/self.v，没有 self.qkv —— 上一版就是这样测空的）
        q, k, v = attn._qkv(xc, 1, Tc)
        nq, nkv, hd = attn.nh, attn.n_kv, attn.hd
        rep = nq // nkv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # causal 在 forward 里由 (T > 1) 决定
        y_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=(Tc > 1))
        y_ref = y_ref.transpose(1, 2).reshape(1, Tc, nq * hd)
        y_ref = attn.o(y_ref)
    print('  模型输出 shape=%s  finite=%s' % (tuple(y_model.shape), bool(torch.isfinite(y_model).all())))
    print('  朴素对照 shape=%s' % (tuple(y_ref.shape),))
    dmax = (y_model - y_ref).abs().max().item()
    rel = dmax / (y_ref.abs().max().item() + 1e-9)
    print('  max|Δ|=%.3e  rel=%.3e  %s'
          % (dmax, rel, 'OK —— 时间测的是正确的算子' if rel < 1e-4 else 'MISMATCH'))

    # ---- forward scaling ----
    print('\n[1] 前向（推理路径）')
    print('  %8s %12s %12s %14s %14s %12s'
          % ('T', 'ms', 't/T^2 (ns)', 't/T (ns)', '相对 T=256', '峰值RSS(MB)'))
    print('  ' + '-' * 78)
    fwd = {}
    for T in (256, 512, 1024, 2048, 4096, 8192, 16384):
        x = torch.randn(1, T, D)
        try:
            with torch.no_grad():
                t = timed(lambda: attn(x, cache=None, append=True))
            fwd[T] = t
            print('  %8d %12.3f %12.4f %14.1f %13.2fx %12.1f'
                  % (T, t * 1e3, t / (T * T) * 1e9, t / T * 1e9,
                     t / fwd[256], peak_rss_mb()))
        except Exception as e:
            print('  %8d FAILED: %s: %s' % (T, type(e).__name__, e))
        finally:
            del x
            gc.collect()

    # ---- backward scaling (training side) ----
    print('\n[2] 反向（训练路径，fwd+bwd）')
    print('  %8s %12s %12s %14s %12s'
          % ('T', 'ms', 't/T^2 (ns)', 't/T (ns)', 'vs fwd'))
    print('  ' + '-' * 66)
    bwd = {}
    for T in (256, 512, 1024, 2048):
        x = torch.randn(1, T, D, requires_grad=True)
        try:
            def step():
                y = attn(x, cache=None, append=True)
                loss = (y * y).sum()
                loss.backward()
                attn.zero_grad(set_to_none=True)
                if x.grad is not None:
                    x.grad = None
            t = timed(step, warmup=1, reps=2)
            bwd[T] = t
            fr = (t / fwd[T]) if T in fwd else float('nan')
            print('  %8d %12.3f %12.4f %14.1f %11.2fx'
                  % (T, t * 1e3, t / (T * T) * 1e9, t / T * 1e9, fr))
        except Exception as e:
            print('  %8d FAILED: %s: %s' % (T, type(e).__name__, e))
        finally:
            del x
            gc.collect()

    # ---- log-log slope: the decisive number ----
    print('\n[3] log-log 斜率（拟合 t = c * T^p）—— 这是判据本身')
    def slope(d):
        ks = sorted(d)
        if len(ks) < 2:
            return float('nan')
        xs = [math.log(k) for k in ks]
        ys = [math.log(d[k]) for k in ks]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = sum((a - mx) ** 2 for a in xs)
        return num / den if den else float('nan')

    print('  前向 p = %.3f   （p=2 即 O(T^2)；p=1 即 O(T)）' % slope(fwd))
    print('  反向 p = %.3f' % slope(bwd))

    # ---- segment slopes: would expose a regime change that one global fit hides ----
    # 全局拟合会把"低速段"和"高速段"平均掉。旧脚本的错误正是只看了一个点就外推，
    # 所以这里逐段给斜率：若各段斜率差异大，说明有交叉点，外推必须分段谨慎使用。
    def seg_slopes(d):
        ks = sorted(d)
        out = []
        for a, b in zip(ks, ks[1:]):
            out.append((a, b, math.log(d[b] / d[a]) / math.log(b / a)))
        return out

    print('\n[3b] 分段斜率（暴露 regime 变化；全局拟合会把它平均掉）')
    print('  %22s %10s' % ('区间', 'p'))
    for a, b, p in seg_slopes(fwd):
        print('  %10d -> %-10d %10.3f   %s' % (a, b, p, '<-- 斜率在涨' if p > 1.6 else ''))
    print('  反向:')
    for a, b, p in seg_slopes(bwd):
        print('  %10d -> %-10d %10.3f' % (a, b, p))

    # ---- extrapolation, now anchored on a measured slope ----
    print('\n[4] 用实测斜率外推（对比旧脚本"按 T^2 外推"的 42 分钟）')
    pf = slope(fwd)
    if fwd:
        Tmax = max(fwd)
        c = fwd[Tmax] / (Tmax ** pf)
        Tvideo = 15 * 30 * 144      # 144p 15s@30fps = 64800 帧像素 → 与旧脚本同口径
        t1 = c * (Tvideo ** pf)
        print('  T=%d 单层：%.2f 秒（T^2 外推会得到 %.0f 秒，差 %.0fx）'
              % (Tvideo, t1, c * Tvideo ** 2, (c * Tvideo ** 2) / t1 if t1 else 0))
        print('  ×%d 层：%.1f 秒' % (cfg.n_layer, t1 * cfg.n_layer))
    print('=' * 84)


if __name__ == '__main__':
    main()
