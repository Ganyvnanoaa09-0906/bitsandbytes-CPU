# -*- coding: utf-8 -*-
"""measure_real_models.py — 不加载权重，只读 safetensors 头，算清每个真实模型在核显上的可行性。

对每个模型给出：
  · 参数量与 dtype（按 key 分类：linear / conv / norm / embed / attention / other）
  · 记一个训练步需要的最小显存（fp32 master + fp16 副本 + 梯度 + 优化器状态）
  · 结论：FIT（可训练）/ INFER-ONLY（只能推理）/ NOT-FIT（连加载都不行）
  · 阻塞点：需要哪种算子还没实现（如 Conv2d）

    python gt1030/measure_real_models.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

LIMIT_MB = 8000          # 多 arena 分块后的可用总量（实测 9 GB / 3 块；单块仍限 4.74 GB）
SINGLE_MB = 4740         # 驱动单 buffer 上限，单个张量不能超过它
OVERHEAD = 1.25          # 激活 + 临时块余量


def classify(key):
    k = key.lower()
    if "conv" in k or "resnet" in k or "downsample" in k:
        return "conv"
    if "attn" in k or "attention" in k or "to_q" in k or "to_k" in k or "to_v" in k:
        return "attention"
    if "norm" in k or "ln_" in k or "gamma" in k or "beta" in k:
        return "norm"
    if "embed" in k or "token" in k or "wte" in k or "shared" in k:
        return "embed"
    if "proj" in k or "linear" in k or k.endswith("weight") and "conv" not in k and "norm" not in k:
        return "linear"
    return "other"


def scan(path):
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    stats = {"params": 0, "bytes": 0, "by_cat": {}, "dtypes": {}, "biggest": ("", 0)}
    for f in files:
        try:
            with safe_open(f, framework="pt", device="cpu") as fh:
                for key in fh.keys():
                    sl = fh.get_slice(key)
                    shape = sl.get_shape()
                    dt = str(sl.get_dtype()).replace("torch.", "")
                    n = 1
                    for d in shape:
                        n *= int(d)
                    bpp = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}.get(dt, 2)
                    nbytes = n * bpp
                    cat = classify(key)
                    stats["params"] += n
                    stats["bytes"] += nbytes
                    stats["by_cat"][cat] = stats["by_cat"].get(cat, 0) + n
                    stats["dtypes"][dt] = stats["dtypes"].get(dt, 0) + nbytes
                    if nbytes > stats["biggest"][1]:
                        stats["biggest"] = (key, nbytes)
        except Exception as error:
            print("    ! %s: %r" % (os.path.basename(f), error))
    stats["files"] = len(files)
    return stats


def verdict(nparam, trainable_frac=1.0):
    """按"全参数训练"估显存：fp32 master 4B + fp16 4B(副本+梯度) + 8-bit 状态 2B。"""
    per = 10 * trainable_frac + 4 * (1 - trainable_frac)   # 冻结部分只留 fp32 master
    need_mb = nparam * per / (1 << 20) * OVERHEAD
    if need_mb <= LIMIT_MB:
        return "FIT", need_mb
    if nparam * 4 / (1 << 20) * 1.3 <= LIMIT_MB:
        return "INFER-ONLY", need_mb
    return "NOT-FIT", need_mb


MODELS = [
    ("Qwen3-1.7B（LLM）", r"D:\work\llm_env\Qwen3-1.7B"),
    ("SD1.5 base（UNet+VAE+TE）", r"D:\work\textmodel\sd15_base"),
    ("SD1.5 VAE 单独", r"D:\work\textmodel\sd15_base\vae"),
    ("SD1.5 UNet 单独", r"D:\work\textmodel\sd15_base\unet"),
    ("Qwen-Image-2.1（LMM/DiT）", r"D:\work\qwen_env\Qwen-Image-2.1"),
    ("Qwen-Image VAE 单独", r"D:\work\qwen_env\Qwen-Image-2.1\vae"),
    ("Qwen-Image Transformer 单独", r"D:\work\qwen_env\Qwen-Image-2.1\transformer"),
    ("Wan2.1-T2V-1.3B", r"D:\work\textmodel\Wan2.1-T2V-1.3B"),
    ("Wan2.1 DiT 单独（如存在）", r"D:\work\textmodel\Wan2.1-T2V-1.3B\dit"),
    ("deepseek-coder-1.3b", r"D:\work\models\models\deepseek-coder-1.3b-base"),
]


def main() -> int:
    print("  显存上限 %d MB（含 %.0f%% 余量）" % (LIMIT_MB, (OVERHEAD - 1) * 100))
    rows = []
    for name, path in MODELS:
        if not os.path.isdir(path):
            print("\n  %-32s 目录不存在" % name)
            continue
        print("\n  %s" % name)
        st = scan(path)
        if st["params"] == 0:
            print("    （没有 safetensors 权重）")
            continue
        n = st["params"]
        cats = ", ".join("%s %.1f%%" % (k, 100.0 * v / n) for k, v in
                         sorted(st["by_cat"].items(), key=lambda kv: -kv[1])[:4])
        v, need = verdict(n)
        print("    参数 %.2f B · 权重 %.1f GB · 文件 %d" % (n / 1e9, st["bytes"] / 1e9, st["files"]))
        print("    构成: %s" % cats)
        print("    训练预估 %.0f MB ⇒ **%s**" % (need, v))
        print("    最大张量: %s (%.0f MB)" % (st["biggest"][0][:60], st["biggest"][1] / 1e6))
        rows.append((name, n, st["bytes"], need, v, st["by_cat"]))
    print("\n  ==== 汇总 ====")
    for name, n, b, need, v, cats in rows:
        print("  %-32s %6.2f B  %7.0f MB  %s" % (name, n / 1e9, need, v))
    conv_heavy = [r[0] for r in rows if r[5].get("conv", 0) > r[1] * 0.2]
    if conv_heavy:
        print("\n  ⚠ 以下模型 >20%% 参数是卷积，需要先实现 Conv2d：")
        for c in conv_heavy:
            print("    - " + c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
