# -*- coding: utf-8 -*-
"""wan_3bit_endtoend.py — 3bit 量化在**真实 30 层模型**上的端到端代价

为什么需要这一步:
    标量量化的 rmsrel 是**逐层**的（4bit 0.092、3bit 0.211、2bit 0.460）。
    但模型有 30 层，误差会**复合**。所以"3bit 能不能用"不能从单层误差推断，
    必须在真实模型上测端到端。

已知锚点（wan_4bit_run.py 实测）:
    4bit（NF4, blocksize 64）: 逐层 0.09237，端到端 rmsrel 0.90（30 层复合）

本脚本:
    用同一套 QuantLinear 框架，把 codebook 换成
      · 4bit NF4（官方表，对照）
      · 3bit 正态分位数（off=0.98，本实验最优）
      · 2bit 正态分位数（off=0.998）
    测每档的 逐层 rmsrel / 端到端 rmsrel / 内存 / 前向时间。

判据（两档，区别对待，避免上一轮把判据放错位置）:
    · 硬判据: 逐层 rmsrel 是否符合该位宽的预期（实现正确性）
    · 报告项: 端到端 rmsrel（复合效应，不作为通过/失败条件）
    · **可判断的实用判据**: 端到端输出的**信噪比**是否还 > 0 dB
      （< 0 dB 意味着输出与 fp32 版本基本不相关 ⇒ 该位宽不可用）
"""
from __future__ import annotations

import ctypes
import gc
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\bitsandbytes")
torch.set_num_threads(int(os.environ.get("THREADS", "6")))

BLOCKSIZE = int(os.environ.get("BLOCKSIZE", "64"))


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("pf", ctypes.c_ulong),
                ("pk", ctypes.c_size_t), ("ws", ctypes.c_size_t)] + \
               [(x, ctypes.c_size_t) for x in "abcdef"]


def peak_gb():
    p = PMC(); p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.pk / 1e9


def codebook(bits, kind="quantile"):
    """目标位宽的码本。4bit 用官方 NF4 表（锚点），低位宽用本实验扫出的最优 offset。"""
    if bits == 4 and kind == "nf4":
        return torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0], dtype=torch.float32)
    # 对称分位数码本（偶数电平、无 0）
    from scipy.stats import norm
    levels = 1 << bits
    half = levels // 2
    off = {2: 0.998, 3: 0.98, 4: 0.9677}.get(bits, 0.9677)
    pos = norm.ppf(torch.linspace(off, 0.5, half + 1)[:-1]).tolist()
    v = sorted([-x for x in pos] + pos)
    t = torch.tensor(v, dtype=torch.float32)
    return t / t.abs().max()


class LowBitLinear(nn.Module):
    """按块做最近邻标量量化，权重存为**原始浮点**（不打包）。
    这里测的是**精度 vs 比特预算**，不是内存布局；打包留给内核层。"""

    def __init__(self, lin, bits, blocksize=64):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.blocksize = blocksize
        cb = codebook(bits, "nf4" if bits == 4 else "quantile")
        self.register_buffer("codebook", cb)
        w = lin.weight.data.float()
        n = w.numel()
        pad = (-n) % blocksize
        flat = w.reshape(-1)
        if pad:
            flat = torch.cat([flat, torch.zeros(pad)])
        blk = flat.reshape(-1, blocksize)
        amax = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        xn = blk / amax
        d = (xn.unsqueeze(-1) - cb.view(1, 1, -1)).abs()
        idx = d.argmin(dim=-1).to(torch.uint8)
        self.register_buffer("idx", idx)
        self.register_buffer("amax", amax)
        self.shape = tuple(w.shape)
        self.n_el = n
        # 逐层误差（量化时即算即弃，不保留原权重）。
        # ⚠️ 用 self.codebook（buffer）而不是局部 cb —— 写法必须与 forward 一致，
        #    否则索引维度对不上（idx 是 2-D (n_blocks, blocksize)，cb 是 1-D）。
        rec = (self.codebook[idx.long()] * amax).reshape(-1)[:n].reshape(self.shape)
        self.layer_rmsrel = ((rec - w).pow(2).mean().sqrt()
                             / (w.pow(2).mean().sqrt() + 1e-30)).item()
        self.bits = bits
        self.has_bias = lin.bias is not None
        if self.has_bias:
            self.register_buffer("bias", lin.bias.data.float().clone())
        self.orig_bytes = n * 4
        # 有效比特：payload + scale(32bit/blocksize)
        self.bits_per_weight = bits + 32.0 / blocksize
        self.quant_bytes = int(n * self.bits_per_weight / 8)
        del w

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        nblk = self.idx.numel()
        rec = (self.codebook[self.idx.long()] * self.amax).reshape(-1)[:self.n_el]
        w = rec.reshape(self.out_features, self.in_features)
        b = self.bias if self.has_bias else None
        return nn.functional.linear(x, w, b)


def replace(root, bits):
    n = 0
    ob = qb = 0
    for mod in root.modules():
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                q = LowBitLinear(child, bits, BLOCKSIZE)
                setattr(mod, cname, q)
                n += 1
                ob += q.orig_bytes
                qb += q.quant_bytes
    return n, ob, qb


def rms_rel(a, b):
    d = a.float() - b.float()
    return (d.pow(2).mean().sqrt() / (a.float().pow(2).mean().sqrt() + 1e-30)).item()


def main():
    print("=" * 90)
    print("低位宽量化在真实 30 层 Wan 模型上的端到端代价")
    print("=" * 90)
    from wan_to_diffusers import build_and_load
    import wan_text_encoder  # noqa  (仅为 sys.path 一致性)
    sd_raw = None
    # 每次都从磁盘重建，保证各档量化的是同一组权重
    print("\n[0] 基准 fp32 前向")
    m, miss, unexp = build_and_load(verbose=False)
    x = torch.randn(1, 16, 5, 16, 16)
    ctx = torch.randn(1, 32, 4096)
    t = torch.tensor([500.0])
    with torch.no_grad():
        y_ref = m(hidden_states=x, encoder_hidden_states=ctx, timestep=t,
                  return_dict=False)[0].clone()
    ref_std = y_ref.std().item()
    print("  missing=%d unexpected=%d  参考输出 std=%.4f  peak=%.2f GB"
          % (len(miss), len(unexp), ref_std, peak_gb()))
    del m
    gc.collect()

    for bits in (4, 3, 2):
        print("\n" + "-" * 90)
        print("[%d-bit] 量化 + 前向" % bits)
        m, _, _ = build_and_load(verbose=False)
        n, ob, qb = replace(m, bits)
        layer = [mm.layer_rmsrel for mm in m.modules()
                 if isinstance(mm, LowBitLinear)]
        bpp = bits + 32.0 / BLOCKSIZE
        print("  替换 %d 层  权重 %.2f -> %.2f GB (%.2f×  payload=%.1f bit/权重)"
              % (n, ob / 1e9, qb / 1e9, ob / qb, bpp))
        print("  逐层 rmsrel: 均值 %.5f  中位 %.5f  最差 %.5f"
              % (statistics.mean(layer), statistics.median(layer), max(layer)))
        with torch.no_grad():
            t0 = time.perf_counter()
            y_q = m(hidden_states=x, encoder_hidden_states=ctx, timestep=t,
                    return_dict=False)[0]
            dt = time.perf_counter() - t0
        e = rms_rel(y_ref, y_q)
        # 端到端信噪比：> 0 dB 表示输出与参考还有正相关
        snr_db = -20 * math.log10(e) if e > 0 else float("inf")
        print("  端到端 rmsrel = %.5f  ⇒ SNR = %+.1f dB   (%.1f s)"
              % (e, snr_db, dt))
        verdict = ("可用（SNR>0dB）" if snr_db > 0 else
                   "不可用（输出与 fp32 不相关）")
        print("  判定: %s" % verdict)
        del m, y_q
        gc.collect()

    print("\n" + "=" * 90)
    print("读法:")
    print("  · 逐层 rmsrel 是硬判据（对照实验：4bit 应 ≈0.092）")
    print("  · 端到端 SNR 是实用判据：<0dB 说明该位宽下模型输出已无意义")
    print("  · 比特预算每权重 = payload + 32/blocksize（scale 不能白给）")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
