# -*- coding: utf-8 -*-
"""diffusion_backends.py — 生图 / 视频训练方法分发层（配合 train_diffusion.py）。

追踪 D 阶段训练方法。除对齐 / RL（align_rl.py）与文本 PEFT（peft_backends.py）外，
本模块覆盖扩散模型的训练入口：

    生图：  lora / ti / dreambooth / full / controlnet / ip_adapter
    视频：  wan_lora / cogvideo_lora / svd_lora / animatediff_lora / video_full

设计约定（与 peft_backends.py 一致）：
  - 本模块只做「注入 / 冻结 / 解冻 / 附加子模型」与 trainable 统计，不写训练循环；
  - 训练循环统一放在 train_diffusion.py（潜变量缓存 / 文本缓存 / 线程策略 / 8bit 优化器）。

diffusers 0.40 已移除内置 Trainer（TextualInversionTrainer / DreamBoothTrainer /
ControlNetTrainer / DDPMTrainer 等），以下均为手写注入逻辑。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn

# 爆改 bnb 仓库根（与 peft_backends.py 同策略，避免 import 到错误版本）
_HERE = os.path.dirname(os.path.abspath(__file__))
_BNB_ROOT = os.path.join(_HERE, "bitsandbytes")
if os.path.isdir(os.path.join(_BNB_ROOT, "bitsandbytes")):
    sys.path.insert(0, _BNB_ROOT)

__all__ = [
    "DiffusionSetup", "apply_method", "METHODS",
    "freeze_module", "unfreeze_module", "count_trainable",
]

# 全部支持的方法（video 类为「结构支持，需模型权重，未实测」）
METHODS = [
    "lora", "ti", "dreambooth", "full", "controlnet", "ip_adapter",
    "wan_lora", "cogvideo_lora", "svd_lora", "animatediff_lora", "video_full",
]

# 默认 LoRA 目标模块（UNet / video transformer 的 attention 投影层通用名）
_DEFAULT_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


@dataclass
class DiffusionSetup:
    """一次方法注入的结果。kind 决定 train_diffusion.py 走哪条训练循环。"""

    method: str
    kind: str                       # "sd_denoise" | "controlnet" | "ip_adapter" | "video"
    trainable_params: int = 0
    total_params: int = 0
    trainable_ratio: float = 0.0
    # extra 里放：placeholder_token_id（ti）/ controlnet（controlnet）/ 状态标记等
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        pct = self.trainable_ratio * 100
        lines = [f"[diffusion:{self.method}] kind={self.kind} "
                 f"trainable={self.trainable_params:,} / total={self.total_params:,} ({pct:.2f}%)"]
        for k, v in self.extra.items():
            if k in ("controlnet", "adapter", "image_encoder", "image_proj", "transformer", "unet"):
                continue  # 子模型对象不打印
            lines.append(f"  {k}={v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def freeze_module(m: nn.Module) -> None:
    m.requires_grad_(False)


def unfreeze_module(m: nn.Module) -> None:
    m.requires_grad_(True)


def count_trainable(modules, exclude: Optional[List[nn.Module]] = None) -> int:
    exclude = exclude or []
    seen = set(id(m) for m in exclude)
    n = 0
    for mod in modules:
        for p in mod.parameters():
            if id(p) == 0 or p in exclude:
                continue
            if p.requires_grad:
                n += p.numel()
    return n


def _total(modules) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def _build_setup(method: str, kind: str, trainable_modules,
                 total_modules, extra: Optional[Dict[str, Any]] = None) -> DiffusionSetup:
    train = count_trainable(trainable_modules)
    total = _total(total_modules)
    return DiffusionSetup(
        method=method, kind=kind,
        trainable_params=train, total_params=total,
        trainable_ratio=(train / total) if total else 0.0,
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# 注入函数（各方法）
# ---------------------------------------------------------------------------

def _apply_lora(unet, rank: int, alpha: int, dropout: float,
                target_modules: Optional[List[str]]) -> DiffusionSetup:
    from peft import LoraConfig, get_peft_model

    freeze_module(unet)
    targets = target_modules or _DEFAULT_TARGETS
    wrapped = get_peft_model(unet, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=targets))
    return _build_setup(
        "lora", "sd_denoise", [wrapped], [wrapped],
        {"target_modules": ",".join(targets), "rank": rank, "unet": wrapped},
    )


def _apply_textual_inversion(text_encoder, tokenizer,
                             placeholder_token: str,
                             initializer_token: str) -> DiffusionSetup:
    """Textual Inversion：新增一个占位 token，只训练它的 embedding 行。

    流程：tokenizer.add_tokens → text_encoder.resize_token_embeddings →
    用 initializer_token 的 embedding 初始化新行 → 冻结其余全部参数。
    """
    if placeholder_token in tokenizer.get_vocab():
        raise ValueError(f"占位 token {placeholder_token!r} 已存在于 tokenizer")

    num_added = tokenizer.add_tokens(placeholder_token)
    if num_added == 0:
        raise ValueError(f"tokenizer.add_tokens({placeholder_token!r}) 未新增任何 token")

    text_encoder.resize_token_embeddings(len(tokenizer))
    # 找到新 token 的 id（通常排在新增的最后）
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # 用 initializer token 的 embedding 初始化新行（不起伏的起点）
    emb = text_encoder.get_input_embeddings()
    init_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if init_ids:
        with torch.no_grad():
            emb.weight.data[placeholder_id] = emb.weight.data[init_ids[0]].clone()

    # 冻结全部，只放行 token embedding（diffusers 官方做法：encoder/norm/pos 全冻结，
    # 只训 token_embedding —— 只有输入里出现的 token 才会累积梯度，占位 token 独占）。
    freeze_module(text_encoder)
    text_encoder.get_input_embeddings().weight.requires_grad_(True)

    return _build_setup(
        "ti", "sd_denoise", [text_encoder], [text_encoder],
        {"placeholder_token": placeholder_token, "placeholder_token_id": placeholder_id,
         "initializer_token": initializer_token},
    )


def _apply_dreambooth(unet, text_encoder, *, train_text_encoder: bool,
                      lora: bool, rank: int, alpha: int, dropout: float,
                      target_modules: Optional[List[str]]) -> DiffusionSetup:
    """DreamBooth：UNet 全参（或 LoRA），可选同时训 text_encoder。

    先验保留（prior preservation）在 train_diffusion.py 的训练循环里处理，这里只管注入。
    """
    trainable = []
    extra = {"train_text_encoder": train_text_encoder, "lora": lora}
    if lora:
        from peft import LoraConfig, get_peft_model
        freeze_module(unet)
        unet = get_peft_model(unet, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
            target_modules=target_modules or _DEFAULT_TARGETS))
        trainable.append(unet)
        extra["unet"] = unet
    else:
        unfreeze_module(unet)
        trainable.append(unet)

    if train_text_encoder:
        unfreeze_module(text_encoder)
        trainable.append(text_encoder)
    else:
        freeze_module(text_encoder)

    return _build_setup(
        "dreambooth", "sd_denoise", trainable, [unet, text_encoder], extra,
    )


def _apply_full(unet, text_encoder, *, train_text_encoder: bool) -> DiffusionSetup:
    trainable = [unet]
    unfreeze_module(unet)
    if train_text_encoder:
        unfreeze_module(text_encoder)
        trainable.append(text_encoder)
    else:
        freeze_module(text_encoder)
    return _build_setup(
        "full", "sd_denoise", trainable, [unet, text_encoder],
        {"train_text_encoder": train_text_encoder},
    )


def _apply_controlnet(unet, controlnet=None, *,
                      conditioning_scale: float = 1.0) -> DiffusionSetup:
    """ControlNet：冻结 UNet，训练（或新建）ControlNetModel。

    若未传入 controlnet，则用 ControlNetModel.from_unet(unet) 原地构造副本。
    兼容 diffusers 0.40：0.6 老 config 的 UNet（如 tiny-sd）mid_block_type 为
    None，from_unet 会因 unknown mid_block_type 报错；这里按 CrossAttn 架构
    修正 config 后再构造（tiny-sd/bk-sdm 均为 CrossAttn 中块，与事实一致）。
    """
    from diffusers import ControlNetModel

    freeze_module(unet)
    if controlnet is None:
        try:
            controlnet = ControlNetModel.from_unet(unet)
        except Exception:
            # 老 config（mid_block_type=None）的 UNet → 兼容构造
            cfg = dict(unet.config)
            controlnet = _from_unet_compat(unet, cfg)
    unfreeze_module(controlnet)
    return _build_setup(
        "controlnet", "controlnet", [controlnet], [controlnet, unet],
        {"controlnet": controlnet, "conditioning_scale": conditioning_scale,
         "status": "controlnet 结构就绪；需 condition 数据（canny/depth/pose）才可训练"},
    )


def _from_unet_compat(unet, cfg):
    """兼容老 config 的 from_unet：按 CrossAttn 架构构造 ControlNet 并复制 UNet 权重。

    tiny-sd/bk-sdm 老 config 的 mid_block_type 为 None（无中块），
    diffusers 0.40 的 ControlNetModel 不接受 None → 用 UNetMidBlock2D(num_layers=0)
    作为**空 mid block**（语义等价于无中块），down_blocks / conv_in / time 正常复制。
    """
    from diffusers import ControlNetModel
    valid = [
        "encoder_hid_dim", "encoder_hid_dim_type", "addition_embed_type",
        "addition_time_embed_dim", "transformer_layers_per_block",
        "in_channels", "flip_sin_to_cos", "freq_shift", "down_block_types",
        "only_cross_attention", "block_out_channels", "layers_per_block",
        "downsample_padding", "mid_block_scale_factor", "act_fn",
        "norm_num_groups", "norm_eps", "cross_attention_dim",
        "attention_head_dim", "num_attention_heads", "use_linear_projection",
        "class_embed_type", "num_class_embeds", "upcast_attention",
        "resnet_time_scale_shift", "projection_class_embeddings_input_dim",
    ]
    kwargs = {k: cfg.get(k) for k in valid if k in cfg}
    kwargs["mid_block_type"] = "UNetMidBlock2D"
    controlnet = ControlNetModel(**kwargs)
    # 复制 UNet 的共享主干权重（conv_in / time / down_blocks / mid_block 存在才复制）
    controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
    controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
    controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
    controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
    if unet.mid_block is not None:
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())
    return controlnet


def _apply_ip_adapter(unet, image_encoder=None, *, num_tokens: int = 4,
                      scale: float = 0.7, status: str = "") -> DiffusionSetup:
    """iP-Adapter：冻结 UNet + 图像编码器，训练图像交叉注意力适配层。

    diffusers 0.40 标准做法（对齐 IPAdapter 训练管线）：
      1. 每个 cross-attention（attn.cross_attention_dim 非空）替换为
         ``IPAdapterAttnProcessor2_0``（内含可训练的 to_k_ip / to_v_ip 投影）；
      2. 新建 ``ImageProjection``（image_embed_dim -> num_tokens x cross_attention_dim）
         把 CLIP 图像特征投到 UNet 的 cross-attn 空间；
      3. 冻结 UNet / image_encoder，只放行 image_proj + 各 processor 的 to_k_ip/to_v_ip。

    image_encoder 缺省时不建 image_proj（最简注入，适配层仍可训练）。
    """
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
    from diffusers.models.embeddings import ImageProjection
    from diffusers.models.attention import Attention

    freeze_module(unet)
    if image_encoder is not None:
        freeze_module(image_encoder)

    cross_attn_dim = getattr(unet.config, "cross_attention_dim", 768) or 768
    processors: List[nn.Module] = []
    n_attn = 0
    for name, m in unet.named_modules():
        # 只注入 cross-attention（attn2；SD UNet 里 attn1 是 self-attn，
        # 其 cross_attention_dim=inner_dim 非 None 但不等同于 cross-attn）
        if not isinstance(m, Attention):
            continue
        if not name.endswith("attn2"):
            continue
        try:
            hidden = getattr(m, "inner_dim", m.query_dim if hasattr(m, "query_dim") else cross_attn_dim)
        except Exception:
            hidden = cross_attn_dim
        proc = IPAdapterAttnProcessor2_0(
            hidden_size=hidden,
            cross_attention_dim=getattr(m, "cross_attention_dim", cross_attn_dim),
            num_tokens=(num_tokens,),
            scale=scale,
        )
        m.set_processor(proc)
        processors.append(proc)
        n_attn += 1

    image_proj = None
    if image_encoder is not None:
        proj_dim = getattr(getattr(image_encoder, "config", None), "projection_dim", None) or 1024
        image_proj = ImageProjection(
            image_embed_dim=proj_dim,
            cross_attention_dim=cross_attn_dim,
            num_image_text_embeds=num_tokens,
        )

    trainable = []
    if image_proj is not None:
        trainable.append(image_proj)
    trainable += processors

    return _build_setup(
        "ip_adapter", "ip_adapter", trainable, [unet] + ([image_encoder] if image_encoder is not None else []),
        {"image_proj": image_proj, "image_encoder": image_encoder,
         "num_tokens": num_tokens, "scale": scale, "attn_count": n_attn,
         "unet": unet,
         "status": status or f"ip_adapter 就绪：{n_attn} 个 cross-attn 注入 IPAdapter processor + image_proj"},
    )


def _apply_video(video_family: str, transformer, *, lora: bool,
                 rank: int, alpha: int, dropout: float,
                 target_modules: Optional[List[str]]) -> DiffusionSetup:
    """视频 LoRA / 全参（Wan / CogVideoX / SVD / AnimateDiff 共用注入逻辑）。

    transformer 为视频降噪骨干（Wan/CogVideoX=transformer，SVD/AnimateDiff=unet(3D)）。
    """
    method = video_family  # "wan_lora" / "cogvideo_lora" / "svd_lora" / "animatediff_lora" / "video_full"
    extra = {"lora": lora, "family": video_family,
             "status": "video 结构就绪；需模型权重（本地暂无 Wan/CogVideoX/SVD/AnimateDiff）"}
    if lora:
        from peft import LoraConfig, get_peft_model
        freeze_module(transformer)
        transformer = get_peft_model(transformer, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
            target_modules=target_modules or _DEFAULT_TARGETS))
        trainable = [transformer]
        extra["transformer"] = transformer
    else:
        unfreeze_module(transformer)
        trainable = [transformer]

    return _build_setup(method, "video", trainable, [transformer], extra)


# ---------------------------------------------------------------------------
# 分发入口
# ---------------------------------------------------------------------------

def apply_method(method: str, *,
                 unet=None, text_encoder=None, tokenizer=None,
                 image_encoder=None, controlnet=None,
                 video_transformer=None,
                 rank: int = 4, alpha: int = 8, dropout: float = 0.1,
                 target_modules: Optional[List[str]] = None,
                 placeholder_token: str = "<concept>",
                 initializer_token: str = "",
                 train_text_encoder: bool = False,
                 conditioning_scale: float = 1.0,
                 num_tokens: int = 4, ip_scale: float = 0.7,
                 **extra) -> DiffusionSetup:
    """分发到具体注入方法。

    Args:
        method: METHODS 之一。
        unet / text_encoder / tokenizer / image_encoder / controlnet / video_transformer:
            按方法传入对应组件。
        target_modules: LoRA 目标模块名列表（缺省用 attention 投影层通用名）。
    """
    if isinstance(target_modules, str):
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]
    m = method.lower().strip()
    if m not in METHODS:
        raise ValueError(f"未知 method: {method!r}，可选 {METHODS}")

    if m == "lora":
        _need(unet, "unet")
        return _apply_lora(unet, rank, alpha, dropout, target_modules)

    if m == "ti":
        _need(text_encoder, "text_encoder"), _need(tokenizer, "tokenizer")
        init = initializer_token or extra.pop("init_token", "") or "photo"
        return _apply_textual_inversion(text_encoder, tokenizer,
                                        placeholder_token, init)

    if m == "dreambooth":
        _need(unet, "unet"), _need(text_encoder, "text_encoder")
        return _apply_dreambooth(unet, text_encoder, train_text_encoder=train_text_encoder,
                                 lora=bool(extra.pop("lora", False)), rank=rank,
                                 alpha=alpha, dropout=dropout, target_modules=target_modules)

    if m == "full":
        _need(unet, "unet"), _need(text_encoder, "text_encoder")
        return _apply_full(unet, text_encoder, train_text_encoder=train_text_encoder)

    if m == "controlnet":
        _need(unet, "unet")
        return _apply_controlnet(unet, controlnet=controlnet,
                                 conditioning_scale=conditioning_scale)

    if m == "ip_adapter":
        _need(unet, "unet")
        return _apply_ip_adapter(unet, image_encoder=image_encoder,
                                 num_tokens=num_tokens, scale=ip_scale)

    # 视频类
    _need(video_transformer, "video_transformer")
    lora = m != "video_full"
    return _apply_video(m, video_transformer, lora=lora, rank=rank,
                        alpha=alpha, dropout=dropout, target_modules=target_modules)


def _need(obj, name: str):
    if obj is None:
        raise ValueError(f"method 需要 {name}，但未传入")
    return obj