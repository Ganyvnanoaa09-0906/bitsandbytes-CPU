# -*- coding: utf-8 -*-
"""qwen_lora_any.py — 通用 LoRA hook 注入（兼容 peft `.lora_A/.lora_B` 与 kohya `.lora.down/up`）。

比 qwen_lightning_bench.LoRAInjector 更通用：
  * 支持任意目标模块（Lightning 只改 128 个 attn；Viggle turbo 改 227 个 linear）；
  * scale 从 safetensors metadata 的 lora_alpha / r 推（缺省 1.0）；
  * 支持 bf16/fp32 权重与任意输入 dtype。
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F


class AnyLoRA:
    def __init__(self, model, lora_path, scale=None, verbose=True):
        from safetensors.torch import load_file
        sd = load_file(lora_path)
        pairs = {}
        for k, v in sd.items():
            if k.endswith(".lora_A.weight"):
                pairs.setdefault(k[: -len(".lora_A.weight")], {})["down"] = v
            elif k.endswith(".lora_B.weight"):
                pairs.setdefault(k[: -len(".lora_B.weight")], {})["up"] = v
            elif k.endswith(".lora.down.weight"):
                pairs.setdefault(k[: -len(".lora.down.weight")], {})["down"] = v
            elif k.endswith(".lora.up.weight"):
                pairs.setdefault(k[: -len(".lora.up.weight")], {})["up"] = v

        if scale is None:
            scale = self._scale_from_meta(lora_path)
        self.scale = float(scale)
        self.handles = []
        self.n = 0
        named = dict(model.named_modules())
        for mod, d in pairs.items():
            if "down" not in d or "up" not in d:
                continue
            key = mod[len("transformer."):] if mod.startswith("transformer.") else mod
            m = named.get(key) or named.get(mod)
            if m is None:
                continue
            A = d["down"].float()
            B = d["up"].float()
            self.handles.append(m.register_forward_hook(self._hook(A, B)))
            self.n += 1
        if verbose:
            print(f"  AnyLoRA: {self.n}/{len(pairs)} 个模块注入（scale={self.scale:.4f}）", flush=True)

    @staticmethod
    def _scale_from_meta(path):
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                md = f.metadata() or {}
            raw = md.get("lora_adapter_metadata")
            if raw:
                cfg = json.loads(raw)
                for k, v in cfg.items():
                    if k.endswith("lora_alpha") and isinstance(v, (int, float)):
                        alpha = float(v)
                        r = None
                        for k2, v2 in cfg.items():
                            if k2.endswith("r") and isinstance(v2, int):
                                r = float(v2)
                        return alpha / r if r else 1.0
        except Exception:
            pass
        return 1.0

    def _hook(self, A, B):
        scale = self.scale

        def hook(module, inputs, output):
            x = inputs[0]
            dt = output.dtype
            delta = F.linear(F.linear(x.float(), A), B) * scale
            return (output.float() + delta).to(dt)

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
