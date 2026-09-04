# -*- coding: utf-8 -*-
"""peft_backends.py — 统一 PEFT 方法分发层（CPU 训练用，集成进爆改 bnb）。

把多种参数高效微调 / 全参微调 / 量化训练方法统一到一个 ``apply_method()`` 入口：

    lora          标准 LoRA（复用 peft LoraConfig）
    qlora         量化基座 + LoRA（复用 quant_lora 融合内核，最快/最省内存）
    p_tuning_v2   深度前缀微调（复用 peft PrefixTuningConfig）
    bitfit        只微调 bias
    vera          共享低秩矩阵 + 可训练缩放向量（复用 peft VeraConfig）
    ia3           权重缩放向量（复用 peft IA3Config）
    full          全参微调
    quant_base    量化基座直接训练（真量化存储 + LSQ 可学习标度，见下方"已知结论"）
    efst          MoE 专家专项微调（复用 efst.py，支持 3D 张量专家）

设计目标：
- 与现有基座（disk_balancer 硬盘均衡 / torch_cpu_kit 线程内存 / bnb 8bit 优化器 /
  8·4bit 量化内核 / GDN 融合内核）无缝配合；
- 每个方法返回 MethodInfo，报告可训练参数、注入层数、内存节省；
- 纯 CPU / AVX2 友好，不偷摸引入 GPU 依赖。

已知结论（来自技术报告 §3.7 / §7）：
- ``quant_base`` 已从旧 QAT（fp32 master 驻留、不减内存）升级为真量化存储 + LSQ：
  基座权重真正存成 8bit/NF4/FP4 码字，fp32 master 不再驻留，仅学习 per-block 标度。
  8bit 省约 75% 权重内存，4bit 省约 87.5%。基座码字本身不更新（需要同时更新基座
  与低秩增量时请用 qlora）。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 爆改 bnb 仓库根（`那很有乐子了~\bitsandbytes\bitsandbytes\` 包），
# 确保 quant_lora / efst 及 peft 检测到的是 CPU 后端版本而非官方 bnb。
_HERE = os.path.dirname(os.path.abspath(__file__))
_BNB_ROOT = os.path.join(_HERE, "bitsandbytes")
if os.path.isdir(os.path.join(_BNB_ROOT, "bitsandbytes")):
    sys.path.insert(0, _BNB_ROOT)

import torch
import torch.nn as nn

__all__ = [
    "MethodInfo",
    "apply_method",
    "METHODS",
    "TEXT_TARGET_MODULES",
]

# 文本模型（LLM）默认注入目标模块
TEXT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
# 仅注意力投影（保守，兼容 MoE / 非标准结构）
ATTN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

METHODS = ("lora", "qlora", "p_tuning_v2", "bitfit", "vera", "ia3",
           "full", "quant_base", "efst")


@dataclass
class MethodInfo:
    """应用某个 PEFT 方法后的结果。"""

    method: str
    trainable_params: int = 0
    total_params: int = 0
    trainable_ratio: float = 0.0
    injected_layers: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        pct = self.trainable_ratio * 100
        lines = [f"[peft:{self.method}] trainable={self.trainable_params:,} "
                 f"/ total={self.total_params:,} ({pct:.2f}%)"]
        if self.injected_layers:
            lines.append(f"  injected_layers={self.injected_layers}")
        for k, v in self.extra.items():
            lines.append(f"  {k}={v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"MethodInfo({self.method}, trainable={self.trainable_params:,}, "
                f"total={self.total_params:,}, ratio={self.trainable_ratio:.4f})")


def _count_params(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _finalize(model: nn.Module, method: str, injected: int = 0,
              extra: Optional[Dict[str, Any]] = None) -> MethodInfo:
    trainable, total = _count_params(model)
    return MethodInfo(
        method=method,
        trainable_params=trainable,
        total_params=total,
        trainable_ratio=trainable / total if total else 0.0,
        injected_layers=injected,
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# 手写方法：BitFit / Full / Quant Base
# ---------------------------------------------------------------------------

def _apply_bitfit(model: nn.Module) -> MethodInfo:
    """只冻结非 bias 参数，保留 bias 可训练。"""
    n_bias = 0
    for name, param in model.named_parameters():
        if "bias" in name or name.endswith(".bias"):
            param.requires_grad_(True)
            n_bias += 1
        else:
            param.requires_grad_(False)
    return _finalize(model, "bitfit", injected=n_bias,
                     extra={"bias_params": n_bias})


def _apply_full(model: nn.Module) -> MethodInfo:
    """全参微调：解冻所有参数。"""
    for param in model.parameters():
        param.requires_grad_(True)
    return _finalize(model, "full")


def _apply_quant_base(model: nn.Module, quant_dtype: str = "nf4",
                      blocksize: Optional[int] = None) -> MethodInfo:
    """量化基座直接训练（真量化存储 + LSQ 可学习标度）。

    把 nn.Linear 的权重真正存成 8bit/NF4/FP4 码字（fp32 master 不再驻留），
    仅学习 per-block 标度 scale。forward 时查表反量化到 fp32 再 GEMM，梯度
    自然流回 scale。主收益是压缩权重内存（8bit 省约 75%，4bit 省约 87.5%）。
    """
    from quant_lora import QuantLinearTrainable

    n = 0
    saved_bytes = 0

    def _replace(module: nn.Module):
        nonlocal n, saved_bytes
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                q = QuantLinearTrainable(
                    child.weight, child.bias, quant_dtype=quant_dtype, blocksize=blocksize)
                orig_bytes = child.in_features * child.out_features * 4
                saved_bytes += orig_bytes - q.quant_weight_bytes()
                setattr(module, name, q)
                n += 1
            else:
                _replace(child)

    _replace(model)
    return _finalize(model, "quant_base", injected=n,
                     extra={"quant_dtype": quant_dtype,
                            "quantized_layers": n,
                            "weight_mem_saved_mb": saved_bytes / (1 << 20)})


# ---------------------------------------------------------------------------
# peft 复用方法：LoRA / QLoRA / P-Tuning v2 / VeRA / IA3
# ---------------------------------------------------------------------------

def _apply_lora(model: nn.Module, r: int = 8, alpha: int = 16,
                target_modules: Optional[List[str]] = None,
                dropout: float = 0.05, bias: str = "none",
                task_type: Optional[str] = None) -> MethodInfo:
    from peft import LoraConfig, get_peft_model
    target = target_modules or ATTN_TARGET_MODULES
    kw = dict(r=r, lora_alpha=alpha, target_modules=target,
              lora_dropout=dropout, bias=bias)
    if task_type:
        kw["task_type"] = task_type
    config = LoraConfig(**kw)
    model = get_peft_model(model, config)
    trainable, total = _count_params(model)
    n_layers = sum(1 for n, m in model.named_modules()
                   if hasattr(m, "lora_A") and hasattr(m, "lora_B"))
    return MethodInfo("lora", trainable, total,
                      trainable / total if total else 0.0,
                      injected_layers=n_layers,
                      extra={"r": r, "alpha": alpha})


def _apply_qlora(model: nn.Module, r: int = 8, alpha: int = 16,
                 quant_dtype: str = "nf4", cache_dequant: bool = True,
                 blocksize: Optional[int] = None,
                 target: type = nn.Linear) -> MethodInfo:
    from quant_lora import quantize_model_8bit_lora
    n, saved = quantize_model_8bit_lora(
        model, lora_r=r, lora_alpha=alpha, cache_dequant=cache_dequant,
        quant_dtype=quant_dtype, blocksize=blocksize, target=target)
    trainable, total = _count_params(model)
    return MethodInfo("qlora", trainable, total,
                      trainable / total if total else 0.0,
                      injected_layers=n,
                      extra={"quant_dtype": quant_dtype,
                             "weight_mem_saved_mb": saved / (1 << 20)})


def _apply_p_tuning_v2(model: nn.Module, num_virtual_tokens: int = 20,
                       encoder_hidden_size: int = 128,
                       task_type: str = "CAUSAL_LM") -> MethodInfo:
    from peft import PrefixTuningConfig, TaskType, get_peft_model
    config = PrefixTuningConfig(
        task_type=getattr(TaskType, task_type, TaskType.CAUSAL_LM),
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
        prefix_projection=False,
    )
    model = get_peft_model(model, config)
    trainable, total = _count_params(model)
    return MethodInfo("p_tuning_v2", trainable, total,
                      trainable / total if total else 0.0,
                      injected_layers=num_virtual_tokens,
                      extra={"num_virtual_tokens": num_virtual_tokens})


def _apply_vera(model: nn.Module, r: int = 256,
                target_modules: Optional[List[str]] = None) -> MethodInfo:
    from peft import VeraConfig, get_peft_model
    target = target_modules or ATTN_TARGET_MODULES
    config = VeraConfig(r=r, target_modules=target)
    model = get_peft_model(model, config)
    trainable, total = _count_params(model)
    n_layers = sum(1 for n, m in model.named_modules()
                   if hasattr(m, "vera_A") and hasattr(m, "vera_B"))
    return MethodInfo("vera", trainable, total,
                      trainable / total if total else 0.0,
                      injected_layers=n_layers, extra={"r": r})


def _apply_ia3(model: nn.Module,
               target_modules: Optional[List[str]] = None,
               feedforward_modules: Optional[List[str]] = None) -> MethodInfo:
    from peft import IA3Config, get_peft_model
    target = target_modules or ATTN_TARGET_MODULES
    feedforward = feedforward_modules or ["gate_proj", "up_proj", "down_proj"]
    config = IA3Config(target_modules=target, feedforward_modules=feedforward)
    model = get_peft_model(model, config)
    trainable, total = _count_params(model)
    n_layers = sum(1 for n, m in model.named_modules() if hasattr(m, "ia3_l"))
    return MethodInfo("ia3", trainable, total,
                      trainable / total if total else 0.0,
                      injected_layers=n_layers or None or 0,
                      extra={"target_modules": target})


def _apply_efst(model: nn.Module, **cfg) -> MethodInfo:
    from efst import EFSTConfig, apply_efst
    _EFST_KEYS = ("expert_indices", "top_k", "calibration_dataloader",
                  "calibration_forward_fn", "num_calibration_batches", "tune_router",
                  "lora", "lora_r", "lora_alpha", "lora_dropout")
    efst_cfg = EFSTConfig()
    for k, v in cfg.items():
        if k in _EFST_KEYS:
            setattr(efst_cfg, k, v)
    info = apply_efst(model, efst_cfg)
    trainable, total = _count_params(model)
    return MethodInfo("efst", trainable, total,
                      trainable / total if total else 0.0,
                      extra={"expert_groups": len(info.groups),
                             "trainable_before": info.trainable_before,
                             "lora_injected": info.lora_injected})


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def apply_method(model: nn.Module, method: str,
                 target_modules: Optional[List[str]] = None,
                 r: int = 8, alpha: int = 16, dropout: float = 0.05,
                 num_virtual_tokens: int = 20,
                 quant_dtype: str = "nf4", cache_dequant: bool = True,
                 blocksize: Optional[int] = None,
                 **extra) -> MethodInfo:
    """把 ``method`` 应用到 ``model``，返回 MethodInfo。

    Args:
        model: 待微调模型（transformers 或任意 nn.Module）。
        method: "lora" / "qlora" / "p_tuning_v2" / "bitfit" / "vera" /
                "ia3" / "full" / "quant_base" / "efst"。
        target_modules: 注入目标模块名列表（lora/qlora/vera/ia3 用）；
            默认文本模型的注意力投影。
        r / alpha: LoRA / VeRA 的秩与缩放。
        num_virtual_tokens: P-Tuning v2 的虚拟 token 数。
        quant_dtype: "nf4" / "fp4" / "8bit"（qlora / quant_base 用）。
        cache_dequant: QLoRA 是否缓存 dequant 结果（True=快，False=省内存）。
        blocksize: 量化块大小（None=自动）。
    """
    m = method.lower().strip()
    if m not in METHODS:
        raise ValueError(f"未知 method: {method!r}，可选 {METHODS}")

    if m == "lora":
        return _apply_lora(model, r=r, alpha=alpha, target_modules=target_modules,
                           dropout=dropout, **extra)
    if m == "qlora":
        return _apply_qlora(model, r=r, alpha=alpha, quant_dtype=quant_dtype,
                            cache_dequant=cache_dequant, blocksize=blocksize, **extra)
    if m == "p_tuning_v2":
        return _apply_p_tuning_v2(model, num_virtual_tokens=num_virtual_tokens, **extra)
    if m == "bitfit":
        return _apply_bitfit(model)
    if m == "vera":
        return _apply_vera(model, r=r, target_modules=target_modules)
    if m == "ia3":
        return _apply_ia3(model, target_modules=target_modules, **extra)
    if m == "full":
        return _apply_full(model)
    if m == "quant_base":
        return _apply_quant_base(model, quant_dtype=quant_dtype, blocksize=blocksize)
    if m == "efst":
        return _apply_efst(model, **extra)
    raise ValueError(f"method {m!r} 未实现")