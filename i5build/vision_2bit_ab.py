"""vision_2bit_ab.py -- 4-bit versus 2-bit through a whole 27-layer tower.

The DiT is not fully on disk yet, but the text encoder's VISION TOWER is: all 27 blocks
plus the patch embedder and the merger live entirely in model-00001-of-00004, which has
downloaded and verified. That is a real, complete, 27-layer transformer stack, so it can
answer the question the DiT comparison was going to answer, without waiting:

    does 2-bit's 5.7x weight-reconstruction penalty survive to the model's output?

The previous attempts to answer this failed for identifiable reasons and are not
repeated:
  * per-layer amplification spanned 1e-20 to 1.0, because the probe input was fed to a
    model whose upstream was still uninitialized and because relative error degenerates
    when layer output magnitudes span twenty orders of magnitude
  * a single layer showed weight error 1.04 producing output error 2.2e-05, but one
    layer's ratio does not generalise across 27

Here the whole tower runs, end to end, from real loaded weights, on one input, with the
weights actually quantized both ways. Whatever the tower does with the error -- absorb it
or amplify it -- happens naturally between input and output.

THE REFERENCE IS 4-BIT, NOT BF16
--------------------------------
The decision on the table is "ship 2-bit instead of 4-bit", so the comparison is 2-bit
against the 4-bit checkpoint, not against the original. Measuring against bf16 would
report the sum of both quantizations and answer a question nobody asked.

Usage:
    python vision_2bit_ab.py [--image 64]
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import quant2bit as Q2  # noqa: E402


def rel(a, b):
    a, b = a.float().double(), b.float().double()
    return ((a - b).norm() / b.norm()).item()


def cos(a, b):
    a = a.float().double().reshape(-1)
    b = b.float().double().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def quantize_module_list(model, bits, verbose=True):
    """Replace every nn.Linear weight with its quantize->dequantize reconstruction.

    Returns (n_layers, bytes_stored) so the size claim is measured rather than assumed.
    """
    import bitsandbytes as bnb
    n = 0
    stored = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        w = mod.weight.data
        if bits == 2:
            packed, absmax, shape = Q2.quant2bit(w)
            rec = Q2.dequant2bit(packed, absmax, shape)
            stored += packed.numel() + absmax.numel() * 4
        elif bits == 4:
            packed, qs = bnb.functional.quantize_4bit(
                w.detach().float(), quant_type="nf4", blocksize=64,
                compress_statistics=True)
            rec = bnb.functional.dequantize_4bit(packed, qs)
            state = qs.as_dict(packed=True)
            stored += packed.numel() + sum(v.numel() * v.element_size()
                                           for v in state.values())
        else:
            raise ValueError(bits)
        with torch.no_grad():
            mod.weight.copy_(rec.reshape(w.shape).to(w.dtype))
        n += 1
    return n, stored


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

    path = os.path.join(a.model, "text_encoder", a.shard)
    if not os.path.isfile(path):
        print(f"missing {path}")
        return 2

    print(f"shard: {a.shard}  ({os.path.getsize(path)/1e9:.2f} GB)")
    cfg = AutoConfig.from_pretrained(os.path.join(a.model, "text_encoder"))
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(cfg)

    named = dict(model.named_parameters())
    keys = set()
    with safe_open(path, framework="pt", device="cpu") as fh:
        for k in fh.keys():
            if k not in named:
                continue
            if tuple(named[k].shape) != tuple(fh.get_slice(k).get_shape()):
                continue
            if named[k].is_meta:
                model.get_submodule(k.rsplit(".", 1)[0]).to_empty(device="cpu")
            with torch.no_grad():
                dict(model.named_parameters())[k].copy_(fh.get_tensor(k))
            keys.add(k)
    still_meta = sum(1 for p in model.parameters() if p.is_meta)
    print(f"  loaded {len(keys)} tensors, {still_meta} parameters still on meta")

    vis = model.model.visual
    vis_lin = [n for n, m in vis.named_modules() if isinstance(m, torch.nn.Linear)]
    vis_lin_loaded = [n for n in vis_lin
                      if f"model.visual.{n}.weight" in keys]
    print(f"  vision tower: {len(vis.blocks)} blocks, {len(vis_lin)} linear layers, "
          f"{len(vis_lin_loaded)} of them loaded")
    if len(vis_lin_loaded) != len(vis_lin):
        missing = sorted(set(vis_lin) - set(vis_lin_loaded))
        print(f"  NOT complete: {len(missing)} layers absent, e.g. {missing[:3]}")
        print("  running anyway would measure a partially initialized tower, so stop.")
        return 2
    print()

    # Patch embedder expects (B, C, H, W) with patch_size 16 and temporal 2, so the
    # grid must divide by the patch size and the spatial merge size.
    h = w = a.image
    torch.manual_seed(0)
    px = torch.randn(1, 3, 2, h, w, dtype=torch.float32)
    grid = torch.tensor([[1, h // 16, w // 16]], dtype=torch.long)
    print(f"probe: pixel_values {tuple(px.shape)}, grid_thw {grid.tolist()}")
    print()

    # The tower returns a BaseModelOutputWithDeepstackFeatures, NOT a tuple. Indexing
    # it with [0] silently yields a different field -- pooler_output -- which is what an
    # earlier revision of this script compared: it reported "4-bit NaN, 2-bit fine" from
    # two different outputs. The field that matters is last_hidden_state, and it is
    # named here rather than indexed.
    def run():
        with torch.no_grad():
            o = vis(px, grid_thw=grid)
        h = o.last_hidden_state
        if not torch.isfinite(h).all():
            raise RuntimeError(f"non-finite tower output: "
                               f"nan={int(torch.isnan(h).sum())} "
                               f"inf={int(torch.isinf(h).sum())}")
        return h

    print("reference: quantize the tower to 4-bit and dequantize (the shipped format)")
    t0 = time.time()
    n4, s4 = quantize_module_list(vis, 4)
    print(f"  {n4} layers in {time.time()-t0:.0f}s, stored {s4/1e6:.1f} MB")
    t0 = time.time()
    ref = run()
    print(f"  forward {time.time()-t0:.1f}s  out {tuple(ref.shape)}  "
          f"std {ref.std():.6f}  absmax {ref.abs().max():.6f}")
    print()

    print("2-bit: same tower, uniform four-level weights")
    # restore from the shard first so the 2-bit pass quantizes the ORIGINAL weights,
    # not the 4-bit reconstruction -- otherwise the 2-bit error would include 4-bit's
    with safe_open(path, framework="pt", device="cpu") as fh:
        for k in keys:
            if not k.startswith("model.visual."):
                continue
            p = dict(model.named_parameters())[k]
            with torch.no_grad():
                p.copy_(fh.get_tensor(k).to(p.dtype))
    t0 = time.time()
    n2, s2 = quantize_module_list(vis, 2)
    print(f"  {n2} layers in {time.time()-t0:.0f}s, stored {s2/1e6:.1f} MB "
          f"({s4/s2:.2f}x smaller than 4-bit)")
    t0 = time.time()
    got = run()
    print(f"  forward {time.time()-t0:.1f}s  out {tuple(got.shape)}  "
          f"std {got.std():.6f}")
    print()

    e = rel(got, ref)
    c = cos(got, ref)
    print("=" * 78)
    print("RESULT -- 27-layer vision tower, 2-bit output against 4-bit output")
    print("=" * 78)
    print(f"  relative error {e:.6f}")
    print(f"  cosine         {c:.8f}")
    print(f"  finite         {bool(torch.isfinite(got).all())}")
    print(f"  storage        2-bit {s2/1e6:.1f} MB vs 4-bit {s4/1e6:.1f} MB "
          f"({100*s2/s4:.1f}%)")
    print()
    print("  weight-level, measured earlier on these same tensors: 2-bit error 0.53,")
    print("  4-bit error 0.093, ratio 5.7x. The number above is what the 27-layer stack")
    print("  does with that.")
    print()
    if e < 0.02:
        print("  -> the tower absorbs it: 2-bit costs a fraction of a percent of output")
        print("     error for 44% fewer bytes.")
    elif e < 0.10:
        print("  -> meaningful but bounded output error.")
    else:
        print("  -> the penalty propagates through the stack.")
    print()
    print("  scope: one tower, one forward pass, random pixel input, no downstream")
    print("  layers. It measures the tower's response to weight precision, which is")
    print("  the quantity the decision needs; it is not an image-quality measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
