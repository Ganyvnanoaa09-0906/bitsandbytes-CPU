"""vae_facts.py -- what the Qwen-Image-2.1 VAE is made of, and what it weighs.

Two numbers for this file are in circulation -- 1.29 GB from the HF API and 1.35 GB
from the file itself -- and the difference matters because the plan is to keep the
VAE unquantized on the grounds that it is small. A 6% discrepancy is easy to wave
away and this is exactly the kind of thing that turns out to be a different file.

Also checks the claim that quantizing it is impossible: bnb has no convolution
quantization, so if the VAE is mostly convolutions (it should be -- the config says
QwenImage21CausalConv3d with is_residual true, decoder_base_dim 144), then only a
small part of it could ever be touched, and it does not need touching anyway.

Usage:
    python vae_facts.py [<vae safetensors path>]
"""
import collections
import json
import os
import sys

DEFAULT = r"D:\work\qwen_env\Qwen-Image-2.1\vae\diffusion_pytorch_model.safetensors"
CFG = r"D:\work\qwen_env\Qwen-Image-2.1\vae\config.json"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from safetensors_head import read_header  # noqa: E402

DT = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2,
      "I8": 1, "U8": 1, "BOOL": 1}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    size_on_disk = os.path.getsize(path)
    print(f"file          : {path}")
    print(f"size on disk  : {size_on_disk:,} B  ({human(size_on_disk)})")
    print()

    meta, hlen = read_header(path)
    total = 0
    dims = collections.Counter()
    dtypes = collections.Counter()
    ndim_bytes = collections.Counter()
    for name, info in meta.items():
        n = 1
        for d in info["shape"]:
            n *= d
        b = n * DT.get(info["dtype"], 0)
        total += b
        dtypes[info["dtype"]] += 1
        ndim_bytes[len(info["shape"])] += b
        # Classify by the parameter name, which is what decides whether
        # bitsandbytes could quantize it: only 2-D *.weight is a Linear.
        if name.endswith(".weight") and len(info["shape"]) == 2:
            dims["linear (2-D .weight)"] += b
        elif len(info["shape"]) == 5:
            dims["conv3d (5-D weight)"] += b
        elif len(info["shape"]) == 4:
            dims["conv2d (4-D weight)"] += b
        elif len(info["shape"]) == 1:
            dims["1-D (norm/bias)"] += b
        else:
            dims[f"{len(info['shape'])}-D other"] += b

    print(f"tensors       : {len(meta)}")
    print(f"declared bytes: {total:,}  ({human(total)})")
    print(f"difference from the file on disk: {size_on_disk - total:+,} B "
          f"(header + alignment)")
    print()
    print("dtype mix:")
    for dt, c in dtypes.most_common():
        print(f"  {dt:6s} {c:4d}")
    print()
    print("what could be quantized, by parameter shape:")
    for k, b in dims.most_common():
        print(f"  {human(b):>10s}  {100*b/max(1,total):5.1f}%  {k}")
    print()
    lin = dims["linear (2-D .weight)"]
    print(f"bnb can quantize only the linear part: {human(lin)} of {human(total)} "
          f"({100*lin/max(1,total):.1f}%)")
    print(f"a 4-bit conversion would save about {human(int(lin*0.72))}, "
          f"taking the VAE from {human(size_on_disk)} to "
          f"{human(size_on_disk - int(lin*0.72))}")
    print()

    if os.path.isfile(CFG):
        with open(CFG, encoding="utf-8") as fh:
            cfg = json.load(fh)
        print("vae/config.json, the fields that say what this architecture is:")
        for k in sorted(cfg):
            v = cfg[k]
            if isinstance(v, (str, int, float, bool)) or v is None:
                print(f"  {k} = {v}")
        print()
        n_layers = sum(1 for k in meta if "CausalConv3d" in k)
        print(f"tensors whose path says CausalConv3d: {n_layers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
