# -*- coding: utf-8 -*-
"""Merge one or more LoRA adapters into a diffusers UNet without peft.

The inference venv (`.venv_dml`, torch 2.4.1 + torch_directml) has **no peft**, and
installing it would risk the pinned torch build, so `pipe.load_lora_weights()` is not an
option there. A LoRA is just a rank-r update, so this merges it directly:

    W += scale * (alpha / r) * (B @ A)

Merging several adapters is a plain sum of their deltas, which is what "stacking LoRAs"
means for a fully-merged (non-adapter) pipeline. Verified: with every scale = 0.0 the
generated frames are byte-identical to a run with no LoRA at all.
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import json
import os

import torch


def _load_pairs(path):
    from safetensors.torch import load_file
    if os.path.isdir(path):
        sd = load_file(os.path.join(path, "adapter_model.safetensors"))
        alpha = r = None
        cfgp = os.path.join(path, "adapter_config.json")
        if os.path.exists(cfgp):
            c = json.load(open(cfgp, encoding="utf-8"))
            alpha, r = c.get("lora_alpha"), c.get("r")
    else:
        sd = load_file(path)
        alpha = r = None
    pairs = {}
    for k, v in sd.items():
        if not k.startswith("base_model.model."):
            continue
        rest = k[len("base_model.model."):]
        if rest.endswith(".lora_A.weight"):
            pairs.setdefault(rest[:-len(".lora_A.weight")], {})["A"] = v
        elif rest.endswith(".lora_B.weight"):
            pairs.setdefault(rest[:-len(".lora_B.weight")], {})["B"] = v
    return pairs, alpha, r


def merge_loras(pipe, paths, scales):
    """Merge `paths` into pipe.unet in order, positionally matched to `scales`."""
    total = 0
    for path, sc in zip(paths, scales):
        pairs, alpha, r = _load_pairs(path)
        scale = sc * ((alpha / r) if (alpha and r) else 1.0)
        mods = dict(pipe.unet.named_modules())
        applied = 0
        for name, ab in pairs.items():
            if "A" not in ab or "B" not in ab or name not in mods:
                continue
            mod = mods[name]
            if not hasattr(mod, "weight"):
                continue
            delta = (ab["B"].float() @ ab["A"].float()) * scale
            with torch.no_grad():
                mod.weight.add_(delta.to(mod.weight.dtype))
            applied += 1
        print(f"[lora] {applied} layers, scale={scale:.3f} (alpha={alpha}, r={r}) <- {path}")
        total += applied
    print(f"[lora] {len(list(paths))} adapter(s) merged; total delta = SUM of all adapters "
          f"({total} layer updates)")
    return total
