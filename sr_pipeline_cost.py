# -*- coding: utf-8 -*-
"""sr_pipeline_cost.py — 超分流水线：Wan 出低分辨率 → SR 放大到 720p/1080p

承接用户思路（这是对的思路）:
    Wan 直出 720p/1080p 已实测不可行（37 TB 注意力矩阵）。
    但超分把「生成分辨率」与「输出分辨率」解耦：**Wan 只出可行包络内的低分辨率，
    再逐帧 4× 放大**。Wan 的 O(T²) 只作用在小 T 上，而 SR 是逐帧卷积、与帧数线性。

上一版 sr_cost.py 在 480 尺寸 + 23 块模型上段错误（两者叠加太大）。
本版修正: **源头固定在 ≤256**（那是 Wan 真正能出的范围），并对每个尺寸单独
try/except + flush，避免一处崩溃丢掉全部数据。

判据:
    SR 一次放大的**单帧时间** × 帧数，与 Wan 生成该低分辨率的时间对比。
    若 SR << Wan 生成 ⇒ 这条路成立。

已知参照（wan_4way_stress.py 实测）:
    T=8192 时单层 attention SDPA 1.598 s ⇒ 30 层 ≈ 48 s / 步
"""
from __future__ import annotations

import gc
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
import sr_cost as S  # noqa: E402


def timed(fn, reps=2):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def fmt(x):
    if x != x:
        return "n/a"
    if x < 60:
        return "%.1f 秒" % x
    if x < 3600:
        return "%.1f 分" % (x / 60)
    if x < 86400:
        return "%.1f 小时" % (x / 3600)
    return "%.1f 天" % (x / 86400)


def main():
    print("=" * 92, flush=True)
    print("超分流水线成本：Wan 低分辨率 → SR 放大", flush=True)
    print("=" * 92, flush=True)

    models = []
    for tag, path, nb in (("anime_6B(6块)", S.ANIME_6B, 6), ("x4plus(23块)", S.X4PLUS, 23)):
        if not os.path.isfile(path):
            continue
        try:
            m, n, miss, unexp = S.load(path, nb)
            print("[load] %-14s %.2fM 参数 missing=%d unexpected=%d"
                  % (tag, n / 1e6, miss, unexp), flush=True)
            models.append((tag, m))
        except Exception as e:
            print("[load] %-14s 失败 %s" % (tag, e), flush=True)

    # ---- 源头固定 ≤256（Wan 可行范围），逐尺寸测 ----
    print("\n[1] 单帧 SR 时间（源头 ≤256，这是 Wan 能出的范围）", flush=True)
    print("  %-14s %10s %12s %12s" % ("模型", "源尺寸", "输出", "单帧 s"), flush=True)
    print("  " + "-" * 54, flush=True)
    table = {}
    for tag, m in models:
        for SZ in (128, 176, 256):
            try:
                x = torch.randn(1, 3, SZ, SZ)
                with torch.no_grad():
                    y = m(x)
                ok = (y.shape[-1] == SZ * 4) and bool(torch.isfinite(y).all())
                t = timed(lambda: m(x), reps=2)
                table[(tag, SZ)] = t
                print("  %-14s %10s %12s %12.2f  %s"
                      % (tag, "%dx%d" % (SZ, SZ), "%dx%d" % (SZ * 4, SZ * 4), t,
                         "" if ok else "⚠ 输出异常"), flush=True)
                del x, y
            except Exception as e:
                print("  %-14s %10d  FAILED %s: %s"
                      % (tag, SZ, type(e).__name__, str(e)[:40]), flush=True)
            gc.collect()

    # ---- 拼出目标分辨率：4× 一次能到多少 ----
    print("\n[2] 一次 4× 放大能覆盖到哪个目标（源头取 Wan 可出的 256）", flush=True)
    for name, H, W in (("480p 854x480", 480, 854), ("720p 1280x720", 720, 1280),
                       ("1080p 1920x1080", 1080, 1920)):
        need_src_h = H / 4
        print("  %-16s 需要源高 %.0f px ⇒ 源 %dx%d %s"
              % (name, need_src_h, int(need_src_h), int(W / 4),
                 "（在 256 内，一次 4x 可达）" if need_src_h <= 256 else
                 "（超出 256 ⇒ 需 2 次 4x，即 16x）"), flush=True)

    # ---- 整段视频的 SR 总成本 ----
    print("\n[3] 整段视频的 SR 总成本（源头 256x256，逐帧）", flush=True)
    print("  %-14s %10s %14s %14s %14s"
          % ("模型", "帧数", "单帧 s", "总时间(6块)", "总时间(23块)"), flush=True)
    print("  " + "-" * 70, flush=True)
    t6 = table.get(("anime_6B(6块)", 256), float("nan"))
    t23 = table.get(("x4plus(23块)", 256), float("nan"))
    for frames in (16, 32, 64, 240):
        print("  %-14s %10d %14.2f %14s %14s"
              % ("256 源", frames, t6, fmt(t6 * frames), fmt(t23 * frames)), flush=True)

    # ---- 与 Wan 生成对比 ----
    print("\n[4] 与 Wan 生成一步对比（实测参照：T=8192 单层 SDPA 1.598 s ⇒ 30 层 ≈ 48 s/步）", flush=True)
    wan_step_8192 = 1.598 * 30
    print("  Wan 一个采样步（T=8192，即 256x256x32帧量级） ≈ %.1f s" % wan_step_8192, flush=True)
    if t6 == t6:
        print("  256x256 单帧 SR（6块） = %.2f s  ⇒  SR 比 Wan 一步便宜 %.1f 倍"
              % (t6, wan_step_8192 / t6), flush=True)
        # 一段 32 帧视频：Wan 生成 N 步 vs SR 32 帧
        for steps in (4, 20):
            tw = wan_step_8192 * steps
            tsr = t6 * 32
            print("    32 帧视频：Wan 生成 %2d 步 = %6.1f s ；SR 32 帧 = %6.1f s ⇒ 合计 %.1f s"
                  % (steps, tw, tsr, tw + tsr), flush=True)

    print("\n" + "=" * 92, flush=True)
    print("结论读法: 若「Wan 生成 + SR」的总时间明显低于「Wan 直出高分辨率」", flush=True)
    print("          （后者已实测为 5-25 天/次前向），则用户思路成立。", flush=True)
    print("=" * 92, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
