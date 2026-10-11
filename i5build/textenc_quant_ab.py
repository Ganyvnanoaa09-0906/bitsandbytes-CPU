"""textenc_quant_ab.py -- does 2-bit weight error actually damage the text embeddings?

What is established: on real weights from this checkpoint, 2-bit uniform quantization
reconstructs with 5.73x the relative error of 4-bit NF4 (0.5338 against 0.0931, mean of
seven sampled tensors including the 151936x4096 embedding).

What that does NOT establish: how much the model's output moves. Weight error and
output error are different quantities, and treating the first as if it were the second
is the same mistake as reading a 10% quantization error as a 10% quality loss. A
diffusion text encoder is a deep nonlinear map; some of that weight error averages out
and some is amplified, and only a measurement says which.

So this quantizes a real layer of the text encoder both ways, runs the same input
through the original and both reconstructions, and reports the output error. The
comparison that matters is 2-bit output error against 4-bit output error -- if the two
outputs are equally close to the original, the extra weight error is being absorbed
and 2-bit costs nothing that can be observed at the output.

Runs on the ONE complete text_encoder shard, since the rest is still downloading.

Usage:
    python textenc_quant_ab.py [--layer N] [--seq 64]
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
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def cos(a, b):
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--component", default="text_encoder")
    ap.add_argument("--shard", default="model-00001-of-00004.safetensors")
    ap.add_argument("--seq", type=int, default=64)
    a = ap.parse_args()

    path = os.path.join(a.model, a.component, a.shard)
    if not os.path.isfile(path):
        print(f"missing shard: {path}")
        return 2

    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )
    from safetensors import safe_open

    print("building the text encoder on the meta device")
    cfg = AutoConfig.from_pretrained(os.path.join(a.model, a.component))
    t0 = time.time()
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(cfg)
    print(f"  built in {time.time()-t0:.1f}s")

    # Load only what is on the shard that is present; leave the rest on meta. The
    # forward pass will therefore run on a partially-initialized model, which is
    # exactly what makes this usable now -- but it means the comparison is of LAYER
    # behaviour, not of end-to-end embeddings. Both sides see the same partial model,
    # so a difference between them is still attributable to the quantization.
    named = dict(model.named_parameters())
    n_loaded = 0
    with safe_open(path, framework="pt", device="cpu") as fh:
        keys = list(fh.keys())
        for k in keys:
            if k not in named:
                continue
            p = named[k]
            if tuple(p.shape) != tuple(fh.get_slice(k).get_shape()):
                continue
            if p.is_meta:
                parent = k.rsplit(".", 1)[0]
                mod = model.get_submodule(parent) if parent else model
                mod.to_empty(device="cpu")
                p = dict(model.named_parameters())[k]
            with torch.no_grad():
                p.copy_(fh.get_tensor(k).to(p.dtype))
            n_loaded += 1
    print(f"  loaded {n_loaded} tensors from {a.shard}; "
          f"{sum(1 for p in model.parameters() if p.is_meta)} remain on meta")

    # Pick a fully-loaded attention block from the vision tower (it is entirely on
    # shard 1) so the layer under test is complete.
    target = None
    for name, mod in model.named_modules():
        if name.endswith("attn.qkv") and isinstance(mod, torch.nn.Linear):
            if not mod.weight.is_meta and mod.weight.shape[0] == 3456:
                target = (name, mod)
                break
    if target is None:
        print("no fully-loaded qkv layer found")
        return 2
    name, lin = target
    print(f"\nlayer under test: {name}  {tuple(lin.weight.shape)}")
    print()

    torch.manual_seed(0)
    x = torch.randn(1, a.seq, lin.in_features, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = lin(x).float()
    print(f"reference output: {tuple(ref.shape)}  "
          f"std {ref.std():.6f}  absmax {ref.abs().max():.6f}")
    print()

    w = lin.weight.data
    results = {}
    for label, rebuild in (
            ("2-bit uniform", lambda ww: Q2.dequant2bit(*Q2.quant2bit(ww)[:2],
                                                        Q2.quant2bit(ww)[2])),
            ("4-bit NF4", Q2.quant4bit_nf4),
    ):
        # quantize then dequantize, and substitute the reconstruction into the layer
        packed, absmax, shape = Q2.quant2bit(w)
        if label.startswith("2-bit"):
            rec = Q2.dequant2bit(packed, absmax, shape)
        else:
            rec = Q2.quant4bit_nf4(w)
        werr = rel(rec, w)
        with torch.no_grad():
            lin.weight.data = rec.to(lin.weight.dtype)
            got = lin(x).float()
            lin.weight.data = w
        oerr = rel(got, ref)
        c = cos(got, ref)
        results[label] = (werr, oerr, c)
        print(f"  {label}")
        print(f"    weight rel err {werr:.6f}")
        print(f"    output rel err {oerr:.6f}   cosine {c:.8f}")
    print()
    r2 = results["2-bit uniform"]
    r4 = results["4-bit NF4"]
    print("comparison:")
    print(f"  weight error  : 2-bit is {r2[0]/max(1e-12,r4[0]):.2f}x the 4-bit error")
    print(f"  OUTPUT error  : 2-bit is {r2[1]/max(1e-12,r4[1]):.2f}x the 4-bit error")
    print()
    print("reading:")
    print("  if the output ratio is far below the weight ratio, the network is")
    print("  absorbing the extra weight error and 2-bit is cheaper than its weight")
    print("  error suggests. If the two ratios are similar, the error propagates and")
    print("  2-bit costs what it looks like it costs.")
    print()
    print("  LIMITATION, stated because it decides how far this generalises: the model")
    print("  is only partly loaded (one shard), so this measures ONE LAYER's response,")
    print("  not the end-to-end embedding. Depth can amplify what one layer absorbs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
