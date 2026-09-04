# -*- coding: utf-8 -*-
"""torch241_compat.py — torch 2.4.1(torch-directml 钉死版本)与 diffusers 0.40 的兼容层。

背景:安装 torch-directml 会把 torch 从 2.13 降到 2.4.1,导致 diffusers 0.40 的
UNet 导入链直接崩溃(SD 训练环境被装坏,与是否启用核显无关)。三处不兼容:

  1. diffusers/models/attention_dispatch.py 用了 `from __future__ import annotations`,
     torch 2.4.1 的 custom op infer_schema 不解析字符串注解 -> ValueError 崩溃;
  2. diffusers 0.40 引用了 torch 2.5+ 的 torch.nn.attention.flex_attention
     (transformer_anyflow_far.py 顶层裸 import);
  3. dispatch_attention_fn 向 F.sdpa 传 torch 2.5+ 才有的 enable_gqa 参数
     (正常路径有兜底,仅显式 dispatch 时触发,桩不处理)。

本模块必须在 `import diffusers` 之前导入。幂等,重复导入无副作用。
torch >= 2.5 环境下所有补丁自动跳过。
"""
import sys
import typing

_TORCH_MIN_INCOMPAT = (2, 5)


def _torch_incompatible() -> bool:
    try:
        import torch
        base = torch.__version__.split("+")[0]
        parts = tuple(int(x) for x in base.split(".")[:3])
        return parts < _TORCH_MIN_INCOMPAT
    except Exception:
        return False


def apply() -> None:
    if getattr(sys, "_torch241_compat_applied", False):
        return
    sys._torch241_compat_applied = True
    if not _torch_incompatible():
        return

    # --- 补丁 1: infer_schema 支持字符串注解(future annotations) ---
    import functools

    import torch._custom_op.impl as _impl

    _orig_infer_schema = _impl.infer_schema

    @functools.wraps(_orig_infer_schema)
    def _infer_schema_patched(func, mutates_args):
        try:
            ann = getattr(func, "__annotations__", None) or {}
            if ann and any(isinstance(v, str) for v in ann.values()):
                hints = typing.get_type_hints(func)
                func.__annotations__ = {k: hints.get(k, v) for k, v in ann.items()}
        except Exception:
            pass
        return _orig_infer_schema(func, mutates_args)

    _impl.infer_schema = _infer_schema_patched

    # --- 补丁 2: flex_attention 运行时桩(仅新模型顶层 import 需要) ---
    import types

    if "torch.nn.attention.flex_attention" not in sys.modules:
        try:
            import torch.nn.attention.flex_attention  # noqa: F401
        except Exception:
            _flex = types.ModuleType("torch.nn.attention.flex_attention")

            class _FlexStub:
                """哑类型:支持 `BlockMask | None` 形式的运行时注解求值。"""

                def __or__(self, other):
                    return object()

                __ror__ = __or__

            _flex.BlockMask = _FlexStub()
            _flex.create_block_mask = None
            sys.modules["torch.nn.attention.flex_attention"] = _flex


apply()
