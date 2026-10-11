"""nan_where.py -- the 4-bit vision tower outputs all-NaN. Where does it start?

vision_2bit_ab.py now raises instead of comparing two different fields, and the raise
says last_hidden_state is 18432 of 18432 elements NaN after 4-bit quantization of the
tower's 116 linear layers.

Two candidates, and they need different fixes:

  A. quantize_4bit produced NaN weights. Then the reconstructed matrices are the
     problem and the forward pass is innocent.
  B. the reconstruction is finite but the forward pass overflows. Then the weights are
     fine and a 27-block stack on this probe input is simply too hot.

The way to tell them apart is to check each layer's reconstruction for finiteness right
after quantizing it, and then to watch the activations block by block during the
forward pass. Blocks are run through a forward hook so the first non-finite activation
is identified by name rather than guessed at.

Usage:
    python nan_where.py [--image 64]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))


def summ(t):
    if t is None:
        return "None"
    t = t.detach().float()
    fin = torch.isfinite(t)
    n_bad = int((~fin).sum())
    mx = t[fin].abs().max().item() if fin.any() else float("nan")
    return (f"shape {tuple(t.shape)} nonfinite {n_bad} absmax {mx:.4e} "
            f"mean {t[fin].mean().item() if fin.any() else float('nan'):+.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--shard", default="model-00001-of-00004.safetensors")
    ap.add_argument("--image", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4, choices=[2, 4])
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
    vis = model.model.visual

    # ---- A: per-layer reconstruction finiteness ----
    print(f"quantizing the vision tower to {a.bits}-bit, checking each layer")
    n_bad = 0
    worst = None
    for name, mod in vis.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        w = mod.weight.data
        if a.bits == 4:
            packed, qs = bnb.functional.quantize_4bit(w.detach().float(),
                                                      quant_type="nf4", blocksize=64,
                                                      compress_statistics=True)
            rec = bnb.functional.dequantize_4bit(packed, qs).reshape(w.shape)
        else:
            import quant2bit as Q2
            packed, absmax, shape = Q2.quant2bit(w)
            rec = Q2.dequant2bit(packed, absmax, shape)
        bad = int((~torch.isfinite(rec)).sum())
        if bad:
            n_bad += 1
            if n_bad <= 6:
                print(f"  {name}: {bad} non-finite of {rec.numel()}")
        mx = rec[torch.isfinite(rec)].abs().max().item() if torch.isfinite(rec).any() else float("nan")
        if worst is None or mx > worst[1]:
            worst = (name, mx, tuple(w.shape))
        with torch.no_grad():
            mod.weight.copy_(rec.reshape(w.shape).to(w.dtype))
    print(f"  layers with non-finite reconstruction: {n_bad}")
    if worst:
        print(f"  largest reconstruction magnitude {worst[1]:.4e} in {worst[0]} {worst[2]}")
    print()

    # ---- B: activation trace, block by block ----
    print("activation trace through the tower")
    seen = []

    def hook(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            seen.append((name, summ(t)))
        return fn

    handles = []
    handles.append(vis.patch_embed.register_forward_hook(hook("patch_embed")))
    for i, blk in enumerate(vis.blocks):
        handles.append(blk.register_forward_hook(hook(f"blocks.{i}")))
    handles.append(vis.merger.register_forward_hook(hook("merger")))

    h = w = a.image
    torch.manual_seed(0)
    px = torch.randn(1, 3, 2, h, w, dtype=torch.float32)
    grid = torch.tensor([[1, h // 16, w // 16]], dtype=torch.long)
    print(f"  probe pixel_values {tuple(px.shape)}, grid_thw {grid.tolist()}")
    print()
    try:
        with torch.no_grad():
            o = vis(px, grid_thw=grid)
        print(f"  forward completed")
        print(f"  last_hidden_state: {summ(o.last_hidden_state)}")
    except Exception as e:  # noqa: BLE001
        print(f"  forward raised: {type(e).__name__}: {str(e)[:160]}")
    finally:
        for hd in handles:
            hd.remove()
    print()
    first_bad = None
    for name, s in seen:
        flag = ""
        if "nonfinite 0" not in s and first_bad is None:
            first_bad = name
            flag = "   <-- first non-finite"
        print(f"  {name:16s} {s}{flag}")
    print()
    print("reading:")
    print("  if the per-layer reconstructions are all finite, A is ruled out and the")
    print("  NaN arises during the forward pass. The trace then names the first block")
    print(f"  that goes non-finite: {first_bad if first_bad else '(none -- all finite)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
