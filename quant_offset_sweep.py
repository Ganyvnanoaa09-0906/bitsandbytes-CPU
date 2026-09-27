# -*- coding: utf-8 -*-
"""quant_offset_sweep.py — 数据驱动地求每个位宽的最优码本

为什么不能直接套用 4-bit 的公式:
    官方 NF4 的构造对 4-bit 是「正 8 / 负 7 / 零 1」。把同一公式推广到 2-bit 时，
    4 个电平会退化成「正 2 / 负 1 / 零 1」——**严重不对称**，负侧实际只有一个电平
    （因为 linspace 在 4 电平下负侧只取到 1 个分位点，归一化后与 -max 重合）。
    实测 NF-2 因此（0.566）反而比均匀栅格（0.511）差。
    ⇒ 这说明**4-bit 的构造不能照搬**，而不是"分位数码本没用"。

本脚本的做法（数据驱动）:
    对每个位宽，扫描 outermost quantile（offset），在**真实权重**上直接量 RMS 相对误差，
    取最优。这等于回答了"该位宽下码本长什么样最优"，而不是套公式。

    offset 越小 ⇒ 覆盖的正态范围越宽 ⇒ 电平铺得越开（适合重尾/饱和）
    offset 越大 ⇒ 范围越窄 ⇒ 电平集中在 0 附近（适合尖峰分布）

判据:
    每个位宽给出最优 offset 与对应的 rmsrel，并与均匀栅格在同比特下对比。
    同时给出**每权重有效比特**（含 scale 开销），因为这才是公平的比较基准。

oracle:
    · 4-bit 时必须能复现官方 NF4 的误差（0.0922），作为实现正确性的锚点；
    · 用两个真实模型（Wan / AnimateDiff adapter）交叉验证，避免只在一边成立。
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\bitsandbytes")
torch.set_num_threads(int(os.environ.get("THREADS", "6")))

from quant_grid_research import (  # noqa: E402
    ADAPTER, WAN, quant_blockwise, rms_rel, uniform_map, read_safetensors)


def quantile_map(bits, offset, symmetric=True):
    """用给定 offset 构造分位数码本。

    symmetric=True: 正负各 levels/2 个（无 0），适合偶数电平
    symmetric=False: 官方 NF4 式（正多一个 + 一个 0）
    """
    from scipy.stats import norm
    levels = 1 << bits
    if symmetric:
        half = levels // 2
        pos = norm.ppf(torch.linspace(offset, 0.5, half + 1)[:-1]).tolist()
        v = sorted([-x for x in pos] + pos)
    else:
        n_pos = levels // 2
        n_neg = levels - 1 - n_pos
        v1 = norm.ppf(torch.linspace(offset, 0.5, n_pos + 1)[:-1]).tolist()
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, n_neg + 1)[:-1])).tolist()
        v = sorted(v1 + [0.0] + v3)
    t = torch.tensor(v, dtype=torch.float32)
    return t / t.abs().max()


def main():
    print("=" * 90)
    print("数据驱动的最优码本：每个位宽扫描 offset")
    print("=" * 90)
    sets = {}
    for tag, path in (("Adapter", ADAPTER), ("Wan", WAN)):
        if os.path.isfile(path):
            ts = [t for t in read_safetensors(path, 40) if t[1].numel() <= 2_000_000]
            sets[tag] = ts
            print("  %-10s %d 张量（≤2M 元素）" % (tag, len(ts)))

    offs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.9677, 0.98, 0.99, 0.995]

    for tag, ts in sets.items():
        print("\n" + "=" * 90)
        print("=== %s（%d 张量）===" % (tag, len(ts)))
        for bits in (2, 3, 4):
            # 对称构造（偶数电平，无 0）
            print("\n  [%d-bit] 对称码本（%d 电平，无 0）" % (bits, 1 << bits))
            print("    %-8s %10s %12s   %s" % ("offset", "bits/权重", "rmsrel", "码本"))
            best = (None, 1e9, None)
            for off in offs:
                try:
                    cb = quantile_map(bits, off, symmetric=True)
                except Exception as e:
                    continue
                errs = []
                b = None
                for k, w in ts:
                    rec, b = quant_blockwise(w, cb, 64)
                    errs.append(rms_rel(w, rec))
                m = statistics.mean(errs)
                if m < best[1]:
                    best = (off, m, cb)
                print("    %-8.4f %10.2f %12.5f   %s"
                      % (off, b, m, " ".join("%+.3f" % x for x in cb.tolist())))
            # 对照：均匀
            cb_u = uniform_map(bits)
            errs = [rms_rel(w, quant_blockwise(w, cb_u, 64)[0]) for k, w in ts]
            mu = statistics.mean(errs)
            print("    %-8s %10.2f %12.5f   %s"
                  % ("UNIFORM", b, mu, " ".join("%+.3f" % x for x in cb_u.tolist())))
            print("    ⇒ 最优 offset=%.4f  rmsrel=%.5f  vs 均匀 %.5f  ⇒ %s"
                  % (best[0], best[1], mu,
                     "分位数赢 %.1f%%" % (100 * (mu - best[1]) / mu) if best[1] < mu
                     else "均匀赢 %.1f%%" % (100 * (best[1] - mu) / best[1])))

    print("\n" + "=" * 90)
    print("读法:")
    print("  · 每个位宽的最优 offset 就是该位宽的**最优分位数码本**")
    print("  · offset 随位宽变化的趋势本身是信息：位宽越低越该覆盖多少分布")
    print("  · 4-bit 若在 offset=0.9677 附近复现实测 0.0922 ⇒ 实现正确（锚点）")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
