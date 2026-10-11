"""sensitivity_map.py -- weight error versus OUTPUT error, measured per layer.

Why this is the measurement that matters
----------------------------------------
Every allocation decision so far has been driven by weight reconstruction error, and
one measurement already shows that quantity can be nearly irrelevant to the model:

    layer model.visual.blocks.0.attn.qkv  (3456, 1152)
      weight relative error 1.0435   (larger than the weights themselves)
      output relative error 0.0000220
      cosine against the original output: 1.00000000

Five orders of magnitude between the two. If that ratio is typical, then ranking layers
by weight error is ranking them by a quantity the network largely ignores, and the
whole 5.7x-versus-4-bit argument says nothing about image quality.

So this measures the conversion factor per layer:

    amplification A = (output relative error) / (weight relative error)

A near zero means the layer is indifferent and can be compressed freely. A large A means
its error reaches the output and it deserves the bits. Only with A in hand does an
allocation become a quality decision instead of a size decision.

How the comparison is kept honest
---------------------------------
Adding noise to a weight and measuring the output change measures the layer's
sensitivity to THAT noise, which is what is wanted -- but the noise has to be the real
quantization error, not synthetic. So each layer is actually quantized at 2, 4 and 8
bits, substituted, and run.

The input matters too: sensitivity depends on the activation distribution, so the same
fixed probe is used for every layer and the probe is reported. A layer can be
insensitive simply because its input is small, and that is a real property, not an
artefact, as long as the probe is stated.

Coverage is limited by the download: only layers whose weights are fully present can be
run. The count of layers measured is reported rather than implied.

Usage:
    python sensitivity_map.py [--seq 32] [--layers 40]
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import quant2bit as Q2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--component", default="text_encoder")
    ap.add_argument("--shard", default="model-00001-of-00004.safetensors")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--layers", type=int, default=40)
    a = ap.parse_args()

    path = os.path.join(a.model, a.component, a.shard)
    if not os.path.isfile(path):
        print(f"missing {path}")
        return 2

    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    print("building the text encoder on the meta device, then loading the shard")
    cfg = AutoConfig.from_pretrained(os.path.join(a.model, a.component))
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(cfg)

    named = dict(model.named_parameters())
    loaded = set()
    with safe_open(path, framework="pt", device="cpu") as fh:
        for k in fh.keys():
            if k not in named:
                continue
            if tuple(named[k].shape) != tuple(fh.get_slice(k).get_shape()):
                continue
            if named[k].is_meta:
                parent = k.rsplit(".", 1)[0]
                model.get_submodule(parent).to_empty(device="cpu")
            with torch.no_grad():
                dict(model.named_parameters())[k].copy_(fh.get_tensor(k))
            loaded.add(k)
    print(f"  loaded {len(loaded)} tensors; "
          f"{sum(1 for p in model.parameters() if p.is_meta)} still on meta")

    # A linear layer is runnable only if its own weight is present; its inputs are
    # synthesised, so nothing upstream needs to exist.
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        wk = f"{name}.weight"
        if wk not in loaded:
            continue
        targets.append((name, mod, wk))
    print(f"  {len(targets)} linear layers with a present weight")
    if not targets:
        print("nothing runnable")
        return 2
    step = max(1, len(targets) // a.layers)
    targets = targets[::step][:a.layers]
    print(f"  measuring {len(targets)} of them")
    print()

    torch.manual_seed(0)
    print(f"probe: N(0,1) input of shape (1, {a.seq}, in_features), bfloat16, "
          f"the same for every layer")
    print()
    print(f"  {'layer':44s} {'shape':>14s} {'A(2bit)':>10s} {'A(4bit)':>10s} "
          f"{'A(8bit)':>10s} {'out std':>10s}")
    rows = []
    for name, mod, wk in targets:
        w = mod.weight.data
        torch.manual_seed(0)
        x = torch.randn(1, a.seq, mod.in_features, dtype=torch.bfloat16)
        with torch.no_grad():
            ref = mod(x).float()
        ostd = float(ref.std())
        if ostd == 0:
            continue
        packed, absmax, shape = Q2.quant2bit(w)
        recs = {
            2: Q2.dequant2bit(packed, absmax, shape),
            4: Q2.quant4bit_nf4(w),
            8: Q2.quant8bit(w),
        }
        A = {}
        for bits, rec in recs.items():
            werr = Q2.rel(rec, w)
            with torch.no_grad():
                mod.weight.data = rec.to(w.dtype)
                got = mod(x).float()
                mod.weight.data = w
            oerr = Q2.rel(got, ref)
            A[bits] = oerr / max(1e-30, werr)
        rows.append((name, tuple(w.shape), A, ostd))
        print(f"  {name[:44]:44s} {str(tuple(w.shape)):>14s} {A[2]:10.3e} "
              f"{A[4]:10.3e} {A[8]:10.3e} {ostd:10.3e}")

    if not rows:
        print("no measurable layers")
        return 2
    print()
    print("=" * 88)
    print("amplification A = output relative error / weight relative error")
    print("=" * 88)
    for bits in (2, 4, 8):
        vals = [r[2][bits] for r in rows]
        print(f"  {bits}-bit: median {st.median(vals):.3e}  min {min(vals):.3e}  "
              f"max {max(vals):.3e}   spread {max(vals)/max(1e-30,min(vals)):.1f}x")
    print()
    print("what this decides:")
    print("  if A is far below 1 for every layer, the model suppresses weight error and")
    print("  the choice between 2 and 4 bits is nearly invisible at the output -- which")
    print("  would also mean the 5.7x weight-error penalty is not a quality penalty.")
    print("  if A varies by orders of magnitude ACROSS layers, allocation has real")
    print("  material: the low-A layers can be compressed and the high-A ones kept.")
    for bits in (2, 4):
        vals = sorted(rows, key=lambda r: r[2][bits])
        print()
        print(f"  {bits}-bit, most sensitive 5 (these would keep their bits):")
        for r in vals[-5:]:
            print(f"    A={r[2][bits]:.3e}  {r[0][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
