"""nan_hunt.py -- where does the vision tower first produce a NaN?

vision_2bit_ab.py produced a NaN output from the 4-bit tower while the 2-bit tower on
the same input was fine. That is backwards from expectation, so before anything is
concluded the NaN has to be located. Candidates, in the order they are tested:

  A. the 4-bit reconstruction itself contains NaN or Inf -- an overflow inside
     quantize_4bit on this layer set
  B. the tower's forward pass overflows with a random N(0,1) pixel input, independent of
     quantization -- the patch embedder's output norm was 1.2e4 in an earlier probe,
     which is a large activation to raise to a 27-layer stack
  C. bf16 accumulation overflows somewhere

Each is separable: A by inspecting the reconstructed weights, B by running the tower with
untouched bf16 weights, C by comparing dtypes.

Usage:
    python nan_hunt.py [--image 64]
"""
import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import quant2bit as Q2  # noqa: E402


def stats(t):
    t = t.detach().float()
    return (f"shape {tuple(t.shape)} finite={bool(torch.isfinite(t).all())} "
            f"nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())} "
            f"absmax={t[torch.isfinite(t)].abs().max().item() if torch.isfinite(t).any() else float('nan'):.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--shard", default="model-00001-of-00004.safetensors")
    ap.add_argument("--image", type=int, default=64)
    a = ap.parse_args()

    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )
    import bitsandbytes as bnb

    path = os.path.join(a.model, "text_encoder", a.shard)
    cfg = AutoConfig.from_pretrained(os.path.join(a.model, "text_encoder"))
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(cfg)

    named = dict(model.named_parameters())
    keys = set()
    raw = {}
    with safe_open(path, framework="pt", device="cpu") as fh:
        for k in fh.keys():
            if k not in named:
                continue
            if tuple(named[k].shape) != tuple(fh.get_slice(k).get_shape()):
                continue
            if named[k].is_meta:
                model.get_submodule(k.rsplit(".", 1)[0]).to_empty(device="cpu")
            t = fh.get_tensor(k)
            raw[k] = t.float()
            with torch.no_grad():
                dict(model.named_parameters())[k].copy_(t)
            keys.add(k)
    print(f"loaded {len(keys)} tensors")

    vis = model.model.visual
    h = w = a.image
    torch.manual_seed(0)
    px = torch.randn(1, 3, 2, h, w, dtype=torch.float32)
    grid = torch.tensor([[1, h // 16, w // 16]], dtype=torch.long)

    # ---- B: the tower with the ORIGINAL weights, no quantization at all ----
    print()
    print("=" * 78)
    print("B) tower with untouched source weights")
    print("=" * 78)
    try:
        with torch.no_grad():
            emb = vis.patch_embed(px)
        print(f"  after patch_embed : {stats(emb)}")
    except Exception as e:  # noqa: BLE001
        print(f"  patch_embed failed: {type(e).__name__}: {str(e)[:120]}")
    try:
        with torch.no_grad():
            out = vis(px, grid_thw=grid)[0]
        print(f"  tower output      : {stats(out)}")
    except Exception as e:  # noqa: BLE001
        print(f"  tower failed: {type(e).__name__}: {str(e)[:120]}")

    # ---- A: does the 4-bit reconstruction contain NaN/Inf? ----
    print()
    print("=" * 78)
    print("A) 4-bit reconstruction of every vision linear weight")
    print("=" * 78)
    bad = []
    worst = None
    for name, mod in vis.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        wk = f"model.visual.{name}.weight"
        if wk not in raw:
            continue
        w0 = raw[wk]
        packed, qs = bnb.functional.quantize_4bit(w0, quant_type="nf4", blocksize=64,
                                                  compress_statistics=True)
        rec = bnb.functional.dequantize_4bit(packed, qs).reshape(w0.shape)
        fin = bool(torch.isfinite(rec).all())
        amax = rec[torch.isfinite(rec)].abs().max().item() if torch.isfinite(rec).any() else float("nan")
        if not fin:
            bad.append((name, tuple(w0.shape), int(torch.isnan(rec).sum()),
                        int(torch.isinf(rec).sum())))
        if worst is None or amax > worst[1]:
            worst = (name, amax, tuple(w0.shape))
    print(f"  layers with non-finite reconstruction: {len(bad)}")
    for b in bad[:8]:
        print(f"    {b[0]}: {b[1]}  nan={b[2]} inf={b[3]}")
    if worst:
        print(f"  largest reconstruction magnitude: {worst[1]:.4e} in {worst[0]} "
              f"{worst[2]}")
    print()

    # ---- the input itself ----
    print("=" * 78)
    print("input and dtype checks")
    print("=" * 78)
    print(f"  pixel_values      : {stats(px)}")
    try:
        with torch.no_grad():
            e = vis.patch_embed(px)
        print(f"  patch_embed out   : std {e.std():.4e} absmax {e.abs().max():.4e} "
              f"dtype {e.dtype}")
    except Exception as ex:  # noqa: BLE001
        print(f"  patch_embed failed: {ex}")
    print()
    print("reading:")
    print("  if B already produces NaN, the NaN is not about quantization at all -- the")
    print("  tower simply overflows on this probe, and a gentler probe is needed.")
    print("  if A shows non-finite reconstructions, quantize_4bit is the source.")
    print("  if neither, the NaN appears only in the quantized forward, which points at")
    print("  accumulation rather than at the weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
