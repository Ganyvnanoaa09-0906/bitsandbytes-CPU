"""How big is the quantized Qwen-Image-2.1 transformer, and how much arithmetic does one
token of it carry? Read straight out of the checkpoint metadata, so the answer does not
depend on loading 3.4 GB of weights.
"""
import json
import os
import re
from collections import Counter

CKPT = r"D:\work\qwen_env\Qwen-Image-2.1-4bit\transformer"
meta = json.load(open(os.path.join(CKPT, "quant_meta.json"), encoding="utf-8"))
layers = meta["layers"]

total = 0
groups = Counter()
for key, info in layers.items():
    n, k = info["shape"]
    total += n * k
    block = key.split(".")[1] if key.startswith("transformer_blocks") else key.split(".")[0]
    groups[block] += n * k

print(f"quantized modules : {len(layers)}")
print(f"quantized params  : {total/1e9:.3f}e9  ({total/1e6:.0f} M)")
print(f"quant_type        : {meta['quant_type']}  blocksize {meta['blocksize']}")
print("\nby group:")
for g, n in groups.most_common(8):
    print(f"  {g:<24} {n/1e6:>9.1f} M")

blocks = sorted({int(k.split(".")[1]) for k in layers if k.startswith("transformer_blocks")})
print(f"\ntransformer blocks: {len(blocks)}")
per_block = Counter()
for key, info in layers.items():
    if key.startswith("transformer_blocks"):
        leaf = re.sub(r"^transformer_blocks\.\d+\.", "", key)
        n, k = info["shape"]
        per_block[leaf] += n * k
print("per-block linears:")
for leaf, n in per_block.most_common():
    print(f"  {leaf:<28} {n/1e6:>8.1f} M")

# One token through the model: 2 FLOPs per parameter actually used.
print(f"\narithmetic per token (2 x params): {2*total/1e9:.2f} GFLOP/token")
for tokens in (4096, 16384):
    print(f"  at {tokens:>6} tokens: {2*total*tokens/1e12:6.2f} TFLOP")
print("\nGB of packed weight read per token (4-bit + fp32 absmax/64 + nested):")
bits = 4 + 32 / meta["blocksize"]
print(f"  ~{bits:.2f} bits/param -> {total*bits/8/1e9:.2f} GB")
