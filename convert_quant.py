# -*- coding: utf-8 -*-
"""HF 模型 → 8bit/4bit 量化格式（分片流式转换，7B 级也安全）。

峰值内存 = 一个 safetensors 分片（7B 分 8 片 ≈ 3.5GB fp32），量化后立即释放。
产物（<out_dir>/）：
  model.safetensors            —— 键名保持原结构，权重换成 {wq, absmax}（JSON meta 描述）
  quant_meta.json              —— 每层 quant_dtype/blocksize/形状
  config.json / tokenizer*     —— 原样复制

用法：
  py -3.11 convert_quant.py --model <模型路径> --out <输出目录> --dtype nf4
  py -3.11 convert_quant.py --model <7B路径> --out <out> --dtype nf4 --blocksize 64

配合 load_quant_model.py 加载（meta device 构建结构，避免 fp32 峰值）。
"""
import argparse
import json
import os
import re
import shutil
import sys

import torch
import safetensors.torch as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes"))
from bitsandbytes.functional import quantize_4bit, quantize_blockwise  # noqa: E402

BLOCKSIZE_DEFAULT = {"8bit": 256, "nf4": 64, "fp4": 64, "nf4u": 64, "fp4u": 64}

# Linear 键（量化对象）：注意力/MLP 投影 + lm_head；embedding/norm 保留原精度
_LINEAR_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$|lm_head\.weight$"
)


def is_linear_key(key: str) -> bool:
    return bool(_LINEAR_RE.search(key))


def quantize_tensor(w: torch.Tensor, dtype: str, blocksize: int):
    """返回 (wq, absmax, code_or_None)。w 是 fp32 CPU 张量。

    nf4u/fp4u（unpacked）：4bit 码解包成 uint8（每码 1 字节），dequant 走 8bit
    标量查表快路径（pshufb 4 平面链路长，慢 ~4 倍）。内存 nf4 的 2 倍（仍省 75%）。
    """
    if dtype == "8bit":
        wq, stats = quantize_blockwise(w.detach().float().reshape(-1), blocksize=blocksize)
        return wq, stats.absmax, None
    if dtype in ("nf4u", "fp4u"):
        qt = dtype[:3]
        wq, stats = quantize_4bit(w.detach().float(), quant_type=qt, blocksize=blocksize)
        # 解包 nibble -> uint8 [n]；顺序必须与 avx2_dequant_4bit 一致：
        # 内核 unpacklo_epi8(hi, lo) -> 输出 [hi, lo, hi, lo, ...]
        packed = wq.reshape(-1)
        lo = (packed & 0x0F).to(torch.uint8)
        hi = (packed >> 4).to(torch.uint8)
        unpacked = torch.stack([hi, lo], dim=1).reshape(-1)
        return unpacked, stats.absmax, stats.code  # code = 16 项 codebook
    wq, stats = quantize_4bit(w.detach().float(), quant_type=dtype, blocksize=blocksize)
    return wq, stats.absmax, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型目录（含 model.safetensors.index.json）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--dtype", choices=["8bit", "nf4", "fp4", "nf4u", "fp4u"], default="nf4",
                    help="nf4u/fp4u = 解包成 uint8（dequant 快 ~4x，内存仍省 75%）")
    ap.add_argument("--blocksize", type=int, default=0, help="0=按 dtype 默认")
    args = ap.parse_args()

    bs = args.blocksize or BLOCKSIZE_DEFAULT[args.dtype]
    os.makedirs(args.out, exist_ok=True)

    # 1) 找分片
    idx_path = os.path.join(args.model, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:  # 单文件
        single = [f for f in os.listdir(args.model) if f.startswith("model") and f.endswith(".safetensors")]
        if not single:
            raise SystemExit("找不到 model.safetensors*")
        shards = single
        weight_map = None

    meta = {"dtype": args.dtype, "blocksize": bs, "layers": {}}
    out_tensors = {}
    processed = 0
    n_quant = 0

    # 2) 逐分片流式量化（每片处理完立即写盘，避免 out_tensors 累积 OOM）
    for shard in shards:
        spath = os.path.join(args.model, shard)
        print(f"[{processed}/{len(shards)}] 加载 {shard} ...", flush=True)
        tensors = st.load_file(spath, device="cpu")  # 峰值 ~一片 fp32
        shard_out = {}
        for key, w in tensors.items():
            if is_linear_key(key) and w.dim() == 2:
                # 量化 Linear 权重（原始可能是 bf16/fp16，先转 fp32）
                wq, absmax, code16 = quantize_tensor(w, args.dtype, bs)
                shard_out[f"{key}.wq"] = wq
                shard_out[f"{key}.absmax"] = absmax
                meta["layers"][key] = {
                    "shape": list(w.shape), "dtype": args.dtype, "blocksize": bs,
                    "bias": f"{key.rsplit('.', 1)[0]}.bias" in tensors,
                }
                if code16 is not None and "code" not in meta:
                    meta["code"] = [float(c) for c in code16.cpu().tolist()]
                n_quant += 1
            else:
                # 非 Linear 权重（embedding/norm/rotary/bias 等）原样保留
                shard_out[key] = w
        del tensors
        # 立即写盘 + 释放（关键：防 OOM）
        out_shard = os.path.join(args.out, f"shard-{processed:05d}.safetensors")
        st.save_file(shard_out, out_shard)
        del shard_out
        processed += 1
        print(f"  分片 {shard} -> {os.path.basename(out_shard)}，累计量化 {n_quant} 层", flush=True)

    # 3) 写 meta + 复制 config/tokenizer
    with open(os.path.join(args.out, "quant_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 4) 复制 config/tokenizer
    for fn in os.listdir(args.model):
        if fn.startswith(("config.json", "tokenizer", "generation_config", "special_tokens", "vocab", "merges")):
            src = os.path.join(args.model, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.out, fn))

    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(args.out) for f in fn) / 2**30
    print(f"\n完成：{args.out}（{sz:.2f} GB，dtype={args.dtype}，blocksize={bs}）")


if __name__ == "__main__":
    main()
