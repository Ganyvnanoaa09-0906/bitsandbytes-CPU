# -*- coding: utf-8 -*-
"""igpu_lora.py — frozen base + LoRA adapters that stay traceable by igpu_fx.

Adapter math is expressed with the ops FX already supports (Linear / Conv2d / add),
so no new device kernel is required:

    y = base(x) + B(A(x))          # A: in -> r, B: r -> out, B initialized to 0

For Conv2d the adapter is a pair of 1x1 convolutions.  The frozen base keeps only
its fp16 copy (see Executor(freeze_base=True, trainable_names=["lora"])).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # B starts at zero, so the wrapped model is exactly the base at step 0.
        return self.base(x) + self.lora_B(self.lora_A(x))


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int = 8):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Conv2d(base.in_channels, r, 1, bias=False)
        self.lora_B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


def wrap_lora(model: nn.Module, r: int = 8, conv: bool = True) -> int:
    """Replace every Linear (and optionally Conv2d) child with a LoRA wrapper."""
    n = 0

    def walk(mod):
        nonlocal n
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, name, LoRALinear(child, r))
                n += 1
            elif conv and isinstance(child, nn.Conv2d):
                setattr(mod, name, LoRAConv2d(child, r))
                n += 1
            else:
                walk(child)

    walk(model)
    return n


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if ".lora_" in n]
