"""weight_sanity.py -- are the DiT's big weight matrices actually non-trivial?

Two readings conflict. norm_q/norm_k come back all zero from the bf16 source, which
is plausible for a model that initializes its QK-norm scales to zero. But
transformer_blocks.0.attn.to_q.weight also reads as identically zero over 16.7M
elements, which is not plausible for a trained transformer -- and its quantized
form is a constant uint8 119, which is what quantizing an all-zero tensor would
produce, so the quantizer agrees with the reading rather than contradicting it.

A tensor that reads as zeros is exactly what a failed or truncated read looks like,
and this session has already produced two diagnostics that lied (a use-after-free
that returned freed memory, and copy_ into a meta tensor that silently did nothing).
So the source is re-read three independent ways and the quantized side is
dequantized to see whether it reproduces the same zeros:

  1. safetensors.load_file on the whole shard, read immediately, nothing deleted
  2. safe_open's get_slice, which is the documented lazy path
  3. a raw read of the file's bytes at the tensor's own offset, parsed by hand

If all three agree the tensor is zero, it is zero. If they disagree, the earlier
reading was an artefact.

Usage:
    python weight_sanity.py
"""
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SRC = r"D:\work\qwen_env\Qwen-Image-2.1"
QUANT = r"D:\work\qwen_env\Qwen-Image-2.1-4bit"

KEYS = [
    "transformer_blocks.0.attn.to_q.weight",
    "transformer_blocks.0.attn.to_k.weight",
    "transformer_blocks.0.attn.to_v.weight",
    "transformer_blocks.0.img_mlp.gate_layer.weight",
    "img_in.weight",
    "transformer_blocks.0.attn.norm_q.weight",
]

DT = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode()), 8 + n


def raw_read(path, key):
    """Parse the tensor straight out of the file, ignoring safetensors entirely."""
    meta, data_start = read_header(path)
    info = meta[key]
    shape = info["shape"]
    n = 1
    for d in shape:
        n *= d
    start, end = info["data_offsets"]
    dt = DT[info["dtype"]]
    with open(path, "rb") as fh:
        fh.seek(data_start + start)
        buf = fh.read(end - start)
    arr = np.frombuffer(buf, dtype=dt)
    return arr, shape, info["dtype"]


def stat(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return (f"n={x.size} min={x.min():+.6g} max={x.max():+.6g} "
            f"absmean={np.abs(x).mean():.6g} zeros={int((x == 0).sum())}")


def main():
    with open(os.path.join(SRC, "transformer",
                           "diffusion_pytorch_model.safetensors.index.json"),
              encoding="utf-8") as fh:
        wm = json.load(fh)["weight_map"]

    # bf16 -> float32 without torch, so the raw path is torch-independent
    def bf16_to_f32(arr):
        u = arr.astype(np.uint32) << 16
        return u.view(np.float32)

    for key in KEYS:
        shard = wm.get(key)
        if shard is None:
            print(f"{key}: not in the index")
            continue
        path = os.path.join(SRC, "transformer", shard)
        print("=" * 74)
        print(f"{key}\n  in {shard}")
        if os.path.getsize(path) < 1_000_000:
            print("  (shard too small to be the real one)")
        # 1 raw
        try:
            arr, shape, dtype = raw_read(path, key)
            f = bf16_to_f32(arr) if dtype == "BF16" else arr.astype(np.float64)
            print(f"  1 raw bytes  : {stat(f)}")
        except Exception as e:  # noqa: BLE001
            print(f"  1 raw bytes  : FAILED {type(e).__name__}: {str(e)[:100]}")

        # 2 load_file, no deletion
        try:
            import safetensors.torch as st
            d = st.load_file(path, device="cpu")
            t = d[key]
            print(f"  2 load_file  : {stat(t.detach().float().numpy())}")
        except Exception as e:  # noqa: BLE001
            print(f"  2 load_file  : FAILED {type(e).__name__}: {str(e)[:100]}")

        # 3 get_slice
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as fh:
                sl = fh.get_slice(key)
                t = sl[:]
            print(f"  3 get_slice  : {stat(t.detach().float().numpy())}")
        except Exception as e:  # noqa: BLE001
            print(f"  3 get_slice  : FAILED {type(e).__name__}: {str(e)[:100]}")
        print()

    # 4 what does the quantized side dequantize to?
    print("=" * 74)
    print("quantized side: dequantize and compare")
    import bitsandbytes as bnb
    sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))
    import qwen21_load as L

    tensors, meta = L.load_checkpoint(QUANT, "transformer")
    for key in KEYS:
        if key not in meta["layers"]:
            print(f"  {key}: not quantized ({'present' if key in tensors else 'absent'})")
            continue
        info = meta["layers"][key]
        qs = bnb.functional.QuantState.from_dict(
            {k[len(key) + 1:]: v for k, v in tensors.items() if k.startswith(key + ".")},
            device=torch.device("cpu"))
        dq = bnb.functional.dequantize_4bit(tensors[key], qs)
        print(f"  {key}\n    dequantized: {stat(dq.detach().float().numpy())} "
              f"(declared shape {info['shape']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
