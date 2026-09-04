# -*- coding: utf-8 -*-
"""加载 convert_quant.py 产出的量化模型（meta device 构建结构，避免 fp32 峰值）。

用法：
    from load_quant_model import load_quant_model
    model = load_quant_model(<量化模型路径>, lora_r=8, lora_alpha=16)
    # model 是 CPU 模型，Linear 已换成 QuantLinearLora（含 LoRA），可直接训练

7B nf4 内存：权重 ~3.5GB + 激活；加载峰值也 ~3.5GB（meta 构建不占内存）。
"""
import json
import os
import sys

import torch
import safetensors.torch as st

from quant_lora import QuantLinearLora


def load_quant_model(quant_dir: str, lora_r: int = 8, lora_alpha: int = 16,
                     cache_dequant: bool = True):
    """加载量化模型。quant_dir 是 convert_quant.py 的输出目录。"""
    with open(os.path.join(quant_dir, "quant_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    # 1) meta device 构建结构（0 内存）。必须在 meta 上下文里创建——
    #    from_config 默认在 CPU 建 8B 随机权重（32GB）会 OOM。
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(quant_dir, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False

    # 2) 读量化权重（多分片文件合并）
    shard_files = sorted(f for f in os.listdir(quant_dir) if f.startswith("shard-") and f.endswith(".safetensors"))
    if not shard_files:  # 兼容旧单文件格式
        shard_files = [f for f in os.listdir(quant_dir) if f.endswith(".safetensors")]
    tensors = {}
    for sf in shard_files:
        tensors.update(st.load_file(os.path.join(quant_dir, sf), device="cpu"))
    print(f"  读入 {len(tensors)} 个张量（来自 {len(shard_files)} 个分片）")

    # 3) 先替换 Linear -> QuantLinearLora（旧 Linear 的 meta 参数被丢弃）
    replaced = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, torch.nn.Linear):
            continue
        wq_key = f"{name}.weight.wq"
        absmax_key = f"{name}.weight.absmax"
        if wq_key not in tensors:
            continue
        out_f, in_f = mod.out_features, mod.in_features
        bias_t = tensors.get(f"{name}.bias")
        code16 = None
        if meta.get("code"):  # nf4u/fp4u 的 16 项 codebook
            code16 = torch.tensor(meta["code"], dtype=torch.float32)
        ql = QuantLinearLora.from_quantized(
            out_f, in_f,
            tensors[wq_key], tensors[absmax_key], bias_t,
            quant_dtype=meta["dtype"], blocksize=meta["blocksize"],
            lora_r=lora_r, lora_alpha=lora_alpha, cache_dequant=cache_dequant,
            code16=code16,
        )
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], ql)
        replaced += 1

    # 4) 非 Linear 权重（embedding/norm 等）从保存的原始权重加载。
    # assign=True 直接赋 fp32 张量（copy 模式会把 fp32 转回模型参数的原 dtype=bf16）
    sd = {k: v.float() for k, v in tensors.items()
          if not (k.endswith(".wq") or k.endswith(".absmax"))}
    model.load_state_dict(sd, strict=False, assign=True)

    # 4.5) 修复残留 meta buffer（rotary inv_freq 等未在权重文件里的）
    theta = float(getattr(config, "rope_theta", 1_000_000.0))
    for name, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        attr = name.rsplit(".", 1)[1]
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        if attr in ("inv_freq", "original_inv_freq"):
            d2 = buf.shape[0]  # 64 = head_dim/2
            freqs = 1.0 / (theta ** (torch.arange(0, d2 * 2, 2, dtype=torch.float32) / (d2 * 2)))
            parent.register_buffer(attr, freqs)
        else:
            parent.register_buffer(attr, torch.zeros(buf.shape, dtype=buf.dtype or torch.float32))

    # 5) LoRA 语义：只训 adapter（base 全冻结）
    for name, p in model.named_parameters():
        if not (name.endswith("lora_A") or name.endswith("lora_B")):
            p.requires_grad_(False)

    print(f"量化模型加载完成：{replaced} 层 QuantLinearLora（{meta['dtype']}, bs={meta['blocksize']}）")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数（LoRA）: {trainable:,}")
    return model
