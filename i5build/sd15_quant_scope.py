"""sd15_quant_scope.py -- what fraction of this UNet can even be quantized?

SD1.5's UNet is a convolutional architecture. bitsandbytes quantizes nn.Linear and
nn.Embedding only -- there is no convolution quantization in the library, which this
project has already measured on another model where 87% of the VAE turned out to be
Conv3d and therefore untouchable.

So before any 2-bit-versus-4-bit comparison on this checkpoint, the scope has to be
stated: how many parameters live in Linear layers, and how many in convolutions. A
comparison that silently quantizes 8% of the weights and reports "2-bit works fine on
SD1.5" would be worthless, and the same comparison reported without the number would be
misleading.

This walks the UNet, classifies every parameter, and reports the split. It also reports
which Linear layers exist, since those are what any quantization experiment here would
actually touch.

Usage:
    python sd15_quant_scope.py [<unet dir>]
"""
import json
import os
import sys

import torch

DEFAULT = r"D:\work\textmodel\sd15_base\unet"


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.isdir(d):
        print(f"not found: {d}")
        return 2

    cfg_path = os.path.join(d, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        print("config.json:")
        for k in ("_class_name", "block_out_channels", "layers_per_block",
                  "attention_head_dim", "cross_attention_dim", "in_channels",
                  "out_channels", "down_block_types", "up_block_types"):
            if k in cfg:
                print(f"  {k} = {cfg[k]}")
        print()

    from diffusers import UNet2DConditionModel

    print(f"loading {d} ...", flush=True)
    model = UNet2DConditionModel.from_pretrained(d, torch_dtype=torch.float32)
    model.eval()

    classes = {}
    total = 0
    for name, mod in model.named_modules():
        own = list(mod.parameters(recurse=False))
        if not own:
            continue
        n = sum(p.numel() for p in own)
        total += n
        cls = type(mod).__name__
        classes[cls] = classes.get(cls, 0) + n

    print(f"total parameters: {total/1e9:.4f}e9 ({human(total*4)} in fp32)")
    print()
    print(f"  {'module class':32s} {'params':>14s} {'share':>8s} {'quantizable':>12s}")
    quant_total = 0
    for cls, n in sorted(classes.items(), key=lambda t: -t[1]):
        can = cls in ("Linear", "Embedding")
        if can:
            quant_total += n
        print(f"  {cls:32s} {n/1e9:13.4f}e9 {100*n/total:7.2f}% "
              f"{'yes' if can else 'no':>12s}")
    print()
    print(f"quantizable by bitsandbytes: {quant_total/1e9:.4f}e9 of {total/1e9:.4f}e9 "
          f"= {100*quant_total/total:.1f}%")
    print(f"not quantizable            : {(total-quant_total)/1e9:.4f}e9 "
          f"= {100*(total-quant_total)/total:.1f}%  (convolutions, norms, biases)")
    print()

    print("the Linear layers a quantization experiment here would actually touch:")
    lines = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            lines.append((name, mod.in_features, mod.out_features,
                          mod.weight.numel()))
    lines.sort(key=lambda t: -t[3])
    for name, i, o, n in lines[:20]:
        print(f"  {name:56s} {i:5d} -> {o:5d}   {n/1e6:8.3f}M")
    if len(lines) > 20:
        print(f"  ... and {len(lines)-20} more")
    print()
    print(f"  {len(lines)} Linear layers, {sum(t[3] for t in lines)/1e6:.1f}M weights")
    print()
    print("reading:")
    print("  this is the scope. If the quantizable share is small, then a 2-bit versus")
    print("  4-bit comparison on this checkpoint measures the attention projections and")
    print("  little else, and any statement about 'SD1.5 at 2-bit' has to carry that")
    print("  number with it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
