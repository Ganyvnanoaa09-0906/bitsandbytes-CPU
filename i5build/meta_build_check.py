"""meta_build_check.py -- can the architectures be built under init_empty_weights?

Before quantizing 32 GB it is worth knowing that the two target architectures can
be instantiated on this transformers/diffusers version with no memory allocated.
Several things could go wrong that have nothing to do with quantization:

  * Qwen3VLForConditionalGeneration may not accept a raw config dict, or may need
    keys the checkpoint's config.json does not carry
  * init_empty_weights may not reach submodules created with plain torch calls
  * the vendored QwenImage21Transformer2DModel may want a diffusers ConfigMixin
    rather than a dict

Each failure would surface as a confusing error in the middle of the quantized
load, so it is cheaper to find out now. This allocates nothing: it counts meta
parameters and reports.

Usage:
    python meta_build_check.py [<model dir>]
"""
import json
import os
import sys
import time

import torch

DEFAULT = r"D:\work\qwen_env\Qwen-Image-2.1"


def report(label, model):
    params = list(model.named_parameters())
    total = sum(p.numel() for _, p in params)
    meta = sum(1 for _, p in params if p.is_meta)
    real = total and sum(p.numel() * p.element_size() for _, p in params if not p.is_meta)
    print(f"  {label}")
    print(f"    parameters      : {len(params)}  ({total/1e9:.3f}e9)")
    print(f"    on the meta dev : {meta} of {len(params)}")
    print(f"    allocated bytes : {real or 0:,} "
          f"({'none -- correct' if not real else 'NOT ZERO -- this defeats the point'})")
    cls = type(model).__name__
    print(f"    class           : {cls}")
    return meta == len(params)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    from accelerate import init_empty_weights

    ok = True
    print(f"model dir: {base}")
    print()

    print("1) QwenImage21Transformer2DModel from transformer/config.json")
    with open(os.path.join(base, "transformer", "config.json"), encoding="utf-8") as fh:
        tcfg = json.load(fh)
    tcfg.pop("_class_name", None)
    tcfg.pop("_diffusers_version", None)
    if isinstance(tcfg.get("axes_dims_rope"), list):
        tcfg["axes_dims_rope"] = tuple(tcfg["axes_dims_rope"])
    try:
        from diffusers.models import QwenImage21Transformer2DModel

        t0 = time.time()
        with init_empty_weights():
            m = QwenImage21Transformer2DModel(**tcfg)
        print(f"    built in {time.time()-t0:.1f}s")
        ok &= report("transformer", m)
        # The attributes the loader relies on.
        print(f"    has img_in: {hasattr(m, 'img_in')}, "
              f"has transformer_blocks: {len(m.transformer_blocks)} blocks")
        blk = m.transformer_blocks[0]
        print(f"    blocks[0].attn.to_out is {type(blk.attn.to_out).__name__} "
              f"with {len(blk.attn.to_out)} entries")
    except Exception as e:  # noqa: BLE001
        print(f"    FAILED {type(e).__name__}: {str(e)[:300]}")
        ok = False
    print()

    print("2) Qwen3VLForConditionalGeneration from text_encoder/config.json")
    try:
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        # A raw dict is rejected on this version: the class wants a PreTrainedConfig
        # instance. AutoConfig dispatches on model_type and builds the nested
        # text_config / vision_config objects the class expects.
        ecfg = AutoConfig.from_pretrained(os.path.join(base, "text_encoder"))
        print(f"    AutoConfig produced {type(ecfg).__name__}")
        t0 = time.time()
        with init_empty_weights():
            m2 = Qwen3VLForConditionalGeneration(ecfg)
        print(f"    built in {time.time()-t0:.1f}s")
        ok &= report("text encoder", m2)
        lm = m2.model.language_model
        print(f"    language_model has {len(lm.layers)} layers")
        print(f"    embed_tokens is {type(lm.embed_tokens).__name__} "
              f"padding_idx={lm.embed_tokens.padding_idx}")
        print(f"    visual tower has {len(m2.model.visual.blocks)} blocks")
        attn = lm.layers[0].self_attn
        print(f"    layers[0].self_attn.q_proj is {type(attn.q_proj).__name__}")
    except Exception as e:  # noqa: BLE001
        print(f"    FAILED {type(e).__name__}: {str(e)[:300]}")
        ok = False
    print()

    print("3) the path the loader takes for each quantized key")
    try:
        from qwen21_policy import kind_of, quantized_module_for

        samples = [
            ("transformer", "transformer_blocks.0.attn.to_out.0.weight"),
            ("transformer", "transformer_blocks.0.attn.to_q.weight"),
            ("transformer", "transformer_blocks.0.img_mlp.gate_layer.weight"),
            ("text_encoder", "model.language_model.layers.0.self_attn.q_proj.weight"),
            ("text_encoder", "model.language_model.embed_tokens.weight"),
        ]
        for comp, s in samples:
            k = kind_of(s)
            _, name = quantized_module_for(k)
            root = m2 if comp == "text_encoder" else m
            try:
                mod = root.get_submodule(s.rpartition(".")[0])
                found = f"found ({type(mod).__name__})"
            except AttributeError:
                found = "NOT FOUND"
            print(f"    {s}")
            print(f"      kind={k} -> {name}, {found}")
    except Exception as e:  # noqa: BLE001
        print(f"    FAILED {type(e).__name__}: {str(e)[:300]}")
        ok = False
    print()

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
