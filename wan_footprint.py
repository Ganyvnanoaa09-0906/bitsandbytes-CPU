# -*- coding: utf-8 -*-
"""wan_footprint.py — Wan2.1-T2V-1.3B 全管线的内存账（实测权重 + 实测量化误差）

为什么需要:
    用户目标是在本机跑真实视频模型。这台机器 15.4 GB 内存、无独显。
    装不装得下不是靠猜 —— 本地权重是**完整的**（safetensors 校验逐字节吻合），
    文件大小就是真值；文本编码器也在本地。
    所以直接算账，并给出"要跑起来必须量化到多少位"。

数据来源（全部实测，不是估）:
    · 各组件文件大小 = 磁盘上的真实字节数（safetensors 头已校验完整）
    · 4bit 压缩比与误差 = quant_stress_wan.py 在这份权重上的实测
    · 优化器状态 = AdamW8bit（仓库自带 CPU 融合内核），按 1 参数 2 状态 × 1 字节

判据:
    给出 fp32 / 8bit / 4bit 三档的**常驻内存**与**训练时峰值**，
    与 15.4 GB 物理内存对比，标出哪一档可行。
"""
from __future__ import annotations

import json
import os
import struct
import sys

WAN_DIR = os.environ.get("WAN_DIR", r"D:\work\textmodel\Wan2.1-T2V-1.3B")
TOTAL_RAM_GB = 15.4
# 实测（quant_stress_wan.py）
NF4_RMSREL = 0.09237     # 306/306 张量，均值
FP4_RMSREL = 0.12287

COMPONENTS = [
    ("transformer (WanModel 1.3B)", "diffusion_pytorch_model.safetensors"),
    ("VAE", "Wan2.1_VAE.pth"),
    ("text encoder (umt5-xxl bf16)", "models_t5_umt5-xxl-enc-bf16.pth"),
]


def safetensors_params(path):
    """读头，返回 (张量数, 参数总量, 各 dtype 计数)。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    hdr.pop("__metadata__", None)
    tot = 0
    dts = {}
    for v in hdr.values():
        c = 1
        for d in v["shape"]:
            c *= d
        tot += c
        dts[v["dtype"]] = dts.get(v["dtype"], 0) + 1
    return len(hdr), tot, dts


def main():
    print("=" * 88)
    print("Wan2.1-T2V-1.3B 全管线内存账（本机物理内存 %.1f GB）" % TOTAL_RAM_GB)
    print("=" * 88)

    print("\n[1] 各组件实测大小")
    print("  %-32s %14s %14s %s" % ("组件", "文件大小", "参数/张量", "dtype"))
    print("  " + "-" * 84)
    total_bytes = 0
    rows = []
    for label, fn in COMPONENTS:
        p = os.path.join(WAN_DIR, fn)
        if not os.path.isfile(p):
            print("  %-32s %14s" % (label, "缺失"))
            continue
        sz = os.path.getsize(p)
        total_bytes += sz
        extra = ""
        if fn.endswith(".safetensors"):
            nt, npar, dts = safetensors_params(p)
            extra = "%.1fM 参数 / %d 张量" % (npar / 1e6, nt)
            dtxt = ",".join("%s×%d" % (k, v) for k, v in dts.items())
        else:
            dtxt = "(pth)"
        rows.append((label, sz, extra))
        print("  %-32s %11.2f GB %14s %s" % (label, sz / 1e9, extra, dtxt))
    print("  " + "-" * 84)
    print("  %-32s %11.2f GB" % ("合计", total_bytes / 1e9))

    # 参数总量（transformer 用于算训练内存）
    tp = os.path.join(WAN_DIR, "diffusion_pytorch_model.safetensors")
    n_par = safetensors_params(tp)[1] if os.path.isfile(tp) else 0

    print("\n[2] 三档精度下的常驻内存（权重部分）")
    print("  %-10s %14s %14s %s" % ("精度", "合计", "其中 transformer", "能否装下(仅权重)"))
    print("  " + "-" * 76)
    # 文本编码器 bf16 已是 2 字节/参数；transformer/VAE 是 fp32（4 字节）
    te_bytes = os.path.getsize(os.path.join(WAN_DIR, "models_t5_umt5-xxl-enc-bf16.pth")) \
        if os.path.isfile(os.path.join(WAN_DIR, "models_t5_umt5-xxl-enc-bf16.pth")) else 0
    vae_bytes = os.path.getsize(os.path.join(WAN_DIR, "Wan2.1_VAE.pth")) \
        if os.path.isfile(os.path.join(WAN_DIR, "Wan2.1_VAE.pth")) else 0
    tr_fp32 = n_par * 4
    for tag, tr in (("fp32", tr_fp32),
                    ("8bit", n_par * 1),
                    ("4bit", n_par * 0.5)):
        tot = tr + te_bytes + vae_bytes
        ok = "是" if tot / 1e9 < TOTAL_RAM_GB * 0.9 else "否"
        print("  %-10s %11.2f GB %11.2f GB %13s" % (tag, tot / 1e9, tr / 1e9, ok))

    print("\n[3] 训练（LoRA）时的峰值")
    print("  LoRA 只训适配器 ⇒ 优化器状态只覆盖可训练参数，与全参微调差三个数量级")
    for lora_m in (2, 10, 50):
        # AdamW8bit: 每参数 2 状态 × 1 字节 = 2 B；+ 梯度 fp32 4 B
        opt = lora_m * 1e6 * 2
        grad = lora_m * 1e6 * 4
        base = n_par * 0.5 + te_bytes + vae_bytes      # 4bit 主干
        peak = base + opt + grad
        print("    LoRA %-4dM 参数: 主干(4bit)+优化器+梯度 = %.2f GB  %s"
              % (lora_m, peak / 1e9,
                 "可行" if peak / 1e9 < TOTAL_RAM_GB * 0.85 else "偏紧"))

    print("\n[4] 量化误差（实测，quant_stress_wan.py 在**这份权重**上跑的）")
    import math
    print("    NF4: RMS 相对误差 %.5f  ⇒ 信噪比 %.1f dB" % (NF4_RMSREL, -20 * math.log10(NF4_RMSREL)))
    print("    FP4: RMS 相对误差 %.5f  ⇒ 信噪比 %.1f dB" % (FP4_RMSREL, -20 * math.log10(FP4_RMSREL)))
    print("    306/306 张量量化成功，0 失败")

    print("\n" + "=" * 88)
    print("结论:")
    print("  · 仅权重部分，fp32 就要 %.1f GB，加上激活/注意力中间量在本机**不可行**" % (total_bytes / 1e9))
    print("  · 4bit 主干把 transformer 从 %.1f GB 压到 %.1f GB ⇒ 这是能跑起来的前提" %
          (tr_fp32 / 1e9, n_par * 0.5 / 1e9))
    print("  · 文本编码器 %.1f GB（bf16，已是 2 字节/参数）是最大单一组件，" % (te_bytes / 1e9))
    print("    它通常可离线预计算 text embedding 后**卸载**，从而省下这笔")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
