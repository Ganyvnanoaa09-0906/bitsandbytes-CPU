"""Which quantized modules carry a `kind`, and which were left to be guessed?

The reference-image path died at `self.pos_embed(interp_indices)` with
"mat1 and mat2 shapes cannot be multiplied (1056x4 and 1152x2304)" because pos_embed is
an nn.Embedding and the loader built a Linear4bit from it. That means the metadata did not
say what kind it was, and the loader defaulted to linear.
"""
import collections
import json

META = r"D:\work\qwen_env\Qwen-Image-2.1-4bit\text_encoder\quant_meta.json"
meta = json.load(open(META, encoding="utf-8"))
layers = meta["layers"]

kinds = collections.Counter(v.get("kind", "<missing>") for v in layers.values())
print("kind distribution:", dict(kinds))
print("top-level meta keys:", sorted(meta))
for k in ("model.language_model.embed_tokens.weight", "model.visual.pos_embed.weight"):
    v = layers.get(k)
    print(f"  {k}\n      kind={v.get('kind', '<missing>')!r}  shape={v['shape']}")

for want in ("embedding", "linear"):
    ks = [k for k, v in layers.items() if v.get("kind") == want]
    print(f"marked {want}: {len(ks)}")
    for k in ks[:3]:
        print("   ", k)
