# -*- coding: utf-8 -*-
"""quant_grid_research.py — 比 4bit 更激进的量化：把 NF4 的构造推广到更低比特

动机（修正我之前的一个不完整结论）:
    我先前测过"2bit 均匀栅格"，得出 SNR 差、且 CPU 上 2bit 解码慢于 4bit。
    但那**只否定了均匀栅格**，没有否定"更低位宽"这件事本身。
    NF4 的核心洞见是：权重近似正态分布，所以**码本应当取正态分位数**，而
    均匀栅格对正态数据是次优的。位宽越低，这个差别越大。

本脚本把 NF4 的构造（functional.py:169 的 create_normal_map）**推广到任意位宽**，
并与其它方案在**真实权重**上对照：

    [A] NF-k    : 正态分位数码本，k bit（k=2,3,4,5）
    [B] UNI-k   : 均匀栅格，同 k（对照，检验"分位数是否真的更好"）
    [C] KM-k    : 每块 k-means 码本（数据自适应，理论最优但需要存码本）
    [D] SVD-r   : 低秩 + 残差量化（正交轴，不是标量量化）

判据（可判定）:
    · **RMS 相对误差**（不用逐元素 rel —— 和接近零时会爆炸，report §10.147.5）
    · **有效比特/权重**（含 scale 与码本开销，不能只算 payload）
    · 与 NF4 4bit **同比特预算**下的优劣

数据:
    · Wan2.1 的真实权重（306 个 2D 张量，本地完整）
    · AnimateDiff 的 motion adapter（另一个真实模型，交叉验证结论不只在 Wan 上成立）

⚠️ 本脚本只测**精度 vs 压缩**。解码速度是另一回事（我已实测 CPU 上 2bit 融合核
   慢于 4bit），会在结论里分开陈述，不混为一谈。
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import time

import torch

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
BLOCKSIZE = int(os.environ.get("BLOCKSIZE", "64"))
MAX_TENSORS = int(os.environ.get("MAX_TENSORS", "60"))   # 抽样，控制时间

WAN = r"D:\work\textmodel\Wan2.1-T2V-1.3B\diffusion_pytorch_model.safetensors"
ADAPTER = r"D:\work\textmodel\animatediff-motion-adapter-v1-5-2\diffusion_pytorch_model.safetensors"


# ---------------------------------------------------------------- 码本
def normal_map(bits, offset=0.9677083):
    """把 NF4 的构造推广到任意位宽：正态分位数，归一化到 [-1,1]。

    ⚠️ 官方 4-bit 的构造是「v1 有 8 个、v3 有 7 个、加 1 个 0」= 16 电平，
       即 **两侧之和 = levels - 1**，正侧比负侧多一个。
       我第一版写成 v1 取 n_pos 个、v3 取 n_pos-1 个，结果 4 电平时得到
       「3 正 + 1 负」（不对称），让 NF-2 的误差（0.566）反而比均匀栅格（0.511）差。
       **那不是分位数码本的问题，是码本构造错了。**

    正确做法：per_side 由 levels 决定 ——
      正侧 = levels//2 个（含 0 之外的最大正值），负侧 = levels - 1 - 正侧 个。
    """
    from scipy.stats import norm
    levels = 1 << bits
    n_pos = levels // 2                 # 正侧电平数（不含 0）
    n_neg = levels - 1 - n_pos          # 负侧电平数（保证两侧+0 = levels）
    v1 = norm.ppf(torch.linspace(offset, 0.5, n_pos + 1)[:-1]).tolist()
    v3 = (-norm.ppf(torch.linspace(offset, 0.5, n_neg + 1)[:-1])).tolist()
    v = sorted(v1 + [0.0] + v3)
    assert len(v) == levels, "码本电平数 %d != %d" % (len(v), levels)
    t = torch.tensor(v, dtype=torch.float32)
    return t / t.abs().max()


def uniform_map(bits):
    levels = 1 << bits
    # 对称均匀：[-1, 1] 上 levels 个电平（含 0 时用奇数个更自然，这里取对称偶数）
    t = torch.linspace(-1, 1, levels)
    return t


# ---------------------------------------------------------------- 量化器
def quant_blockwise(w, codebook, blocksize=64):
    """按块做最近邻标量量化。返回 (反量化重建, 每权重有效比特)。

    有效比特 = payload(bits) + scale(32 bits / blocksize) [+ 码本开销由调用方加]
    """
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (-n) % blocksize
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blk = flat.reshape(-1, blocksize)
    amax = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    xn = blk / amax                                   # [-1, 1]
    # 最近邻
    d = (xn.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
    idx = d.argmin(dim=-1)
    rec = codebook[idx] * amax
    rec = rec.reshape(-1)[:n].reshape(w.shape)
    bits = math.log2(len(codebook))
    overhead = 32.0 / blocksize
    return rec, bits + overhead


def quant_kmeans(w, bits, blocksize=64, iters=8):
    """每块独立 k-means 码本（数据自适应）。返回 (重建, 有效比特)。

    码本开销：每块 2^bits 个 fp16 电平 ⇒ (levels*16)/blocksize 比特/权重。
    """
    from scipy.cluster.vq import kmeans2
    import numpy as np
    flat = w.reshape(-1).numpy()
    n = flat.size
    pad = (-n) % blocksize
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blk = flat.reshape(-1, blocksize)
    levels = 1 << bits
    out = np.empty_like(blk)
    for i in range(blk.shape[0]):
        x = blk[i]
        try:
            cb, lab = kmeans2(x.astype(np.float64), levels, minit="++", iter=iters,
                              seed=0)
            out[i] = cb[lab]
        except Exception:
            out[i] = x
    rec = torch.from_numpy(out.reshape(-1)[:n].reshape(tuple(w.shape))).float()
    payload = bits + 32.0 / blocksize + (levels * 16) / blocksize
    return rec, payload


def lowrank_plus(w, rank_frac=0.25, bits_res=4, blocksize=64):
    """低秩 + 残差量化（正交于标量量化的另一条轴）。"""
    m, n = w.shape
    r = max(1, int(min(m, n) * rank_frac))
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
    approx = (U_r * S_r) @ Vh_r
    res = w - approx
    cb = normal_map(bits_res)
    res_q, b_res = quant_blockwise(res, cb, blocksize)
    # 比特：低秩部分 r*(m+n) 个 fp16 + 残差的量化比特
    bits_lr = (r * (m + n) * 16) / (m * n)
    return approx + res_q, bits_lr + b_res


def rms_rel(a, b):
    d = (a.float() - b.float())
    return (d.pow(2).mean().sqrt() / (a.float().pow(2).mean().sqrt() + 1e-30)).item()


# ---------------------------------------------------------------- 数据
def read_safetensors(path, limit):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        meta = {k: v for k, v in hdr.items() if k != "__metadata__"}
        tensors = []
        for k, v in meta.items():
            if len(v["shape"]) != 2 or not k.endswith("weight"):
                continue
            f.seek(base + v["data_offsets"][0])
            raw = f.read(v["data_offsets"][1] - v["data_offsets"][0])
            dt = {"F32": torch.float32, "BF16": torch.bfloat16,
                  "F16": torch.float16}[v["dtype"]]
            tensors.append((k, torch.frombuffer(bytearray(raw), dtype=dt)
                            .reshape(v["shape"]).float().clone()))
            if len(tensors) >= limit:
                break
    return tensors


def main():
    print("=" * 92)
    print("更激进的量化：NF4 构造推广到低位宽 vs 均匀栅格 vs 数据自适应码本")
    print("=" * 92)
    print("blocksize=%d  抽样张量数=%d" % (BLOCKSIZE, MAX_TENSORS))

    sets = {}
    for tag, path in (("Wan2.1-1.3B", WAN), ("AnimateDiff-adapter", ADAPTER)):
        if os.path.isfile(path):
            try:
                sets[tag] = read_safetensors(path, MAX_TENSORS)
                print("  %-20s %d 个 2D 权重张量" % (tag, len(sets[tag])))
            except Exception as e:
                print("  %-20s 读取失败 %s" % (tag, e))

    # 码本一览
    print("\n[码本]")
    for bits in (2, 3, 4):
        nm = normal_map(bits)
        um = uniform_map(bits)
        print("  %d-bit  NOFM: %s" % (bits, " ".join("%.4f" % x for x in nm.tolist())))
        print("  %d-bit  UNIF: %s" % (bits, " ".join("%.4f" % x for x in um.tolist())))

    # ---- 主对照 ----
    print("\n[精度 vs 有效比特]  每个方案在全部张量上的 RMS 相对误差（均值）")
    schemes = []
    for bits in (2, 3, 4, 5):
        schemes.append(("NF-%d" % bits, "nomap", bits))
    for bits in (2, 3, 4):
        schemes.append(("UNI-%d" % bits, "uni", bits))
    for bits in (2, 3):
        schemes.append(("KM-%d" % bits, "km", bits))
    schemes.append(("SVD25+NF4", "svd", None))

    for tag, tensors in sets.items():
        print("\n  === %s（%d 张量）===" % (tag, len(tensors)))
        print("  %-12s %10s %12s %12s" % ("方案", "有效比特/权重", "rmsrel 均值", "rmsrel 中位"))
        print("  " + "-" * 52)
        for name, kind, bits in schemes:
            errs = []
            bpp = None
            t0 = time.perf_counter()
            for k, w in tensors:
                if w.numel() > 4_000_000:      # 跳过超大张量，控制时间
                    continue
                if kind == "nomap":
                    rec, b = quant_blockwise(w, normal_map(bits), BLOCKSIZE)
                elif kind == "uni":
                    rec, b = quant_blockwise(w, uniform_map(bits), BLOCKSIZE)
                elif kind == "km":
                    rec, b = quant_kmeans(w, bits, BLOCKSIZE)
                else:
                    rec, b = lowrank_plus(w, 0.25, 4, BLOCKSIZE)
                errs.append(rms_rel(w, rec))
                bpp = b
            if not errs:
                continue
            import statistics
            dt = time.perf_counter() - t0
            print("  %-12s %10.2f %12.5f %12.5f   (%.1f s)"
                  % (name, bpp, statistics.mean(errs), statistics.median(errs), dt))

    print("\n" + "=" * 92)
    print("判据读法:")
    print("  · NF-k 优于 UNI-k  ⇒ 正态分位数码本确实更适合权重（位宽越低差距应越大）")
    print("  · KM-k 接近 NF-k 但比特更高 ⇒ 数据自适应码本的额外开销不值")
    print("  · 同比特下谁误差最低，就是该位宽下的正确选择")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
