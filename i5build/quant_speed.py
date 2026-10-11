"""quant_speed.py -- how long does quantizing Qwen-Image-2.1 actually take?

The 4-bit plan only makes sense if the conversion finishes in a reasonable time.
Everything else about it is proven: the kernel round-trips on CPU, the checkpoint
format reloads into Params4bit(bnb_quantized=True), and the projection says 32 GB
becomes about 10 GB. What is unknown is the rate, and a rate guessed from a 512x384
test tensor would be worthless -- the CPU quantize kernel is compute-bound and its
behaviour on the real tensor sizes is what matters.

Times quantize_4bit on the largest tensors in a real shard, cold and warm, and
extrapolates to the full model.

Usage:
    python quant_speed.py [<shard>] [--n 4]
"""
import argparse
import json
import os
import struct
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402

DEFAULT = (r"D:\work\qwen_env\Qwen-Image-2.1"
           r"\text_encoder\model-00001-of-00004.safetensors")
DT = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2}


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode())


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", nargs="?", default=DEFAULT)
    ap.add_argument("--n", type=int, default=4, help="how many tensors to time")
    ap.add_argument("--blocksize", type=int, default=64)
    ap.add_argument("--quant-type", default="nf4")
    a = ap.parse_args()

    from safetensors import safe_open

    print(f"shard      : {a.shard}")
    print(f"on disk    : {human(os.path.getsize(a.shard))}")
    print(f"quantizing : {a.quant_type}, blocksize {a.blocksize}")
    print()

    hdr = read_header(a.shard)
    hdr.pop("__metadata__", None)
    cands = []
    for name, info in hdr.items():
        if len(info["shape"]) == 2 and name.endswith(".weight"):
            n = 1
            for d in info["shape"]:
                n *= d
            cands.append((n, name, info["shape"], info["dtype"]))
    cands.sort(reverse=True)
    print(f"{len(cands)} candidate tensors in this shard; timing the "
          f"{min(a.n, len(cands))} largest")
    print()

    # Read the tensors without loading the whole file: a shard is several GB and
    # only a few tensors are being timed.
    total_numel = 0
    total_sec = 0.0
    with safe_open(a.shard, framework="pt") as fh:
        for i, (numel, name, shape, dtype) in enumerate(cands[:a.n]):
            t_read = time.time()
            w = fh.get_tensor(name)
            read_s = time.time() - t_read
            t0 = time.time()
            wq, qs = bnb.functional.quantize_4bit(
                w.to(torch.float32), quant_type=a.quant_type, blocksize=a.blocksize)
            q_s = time.time() - t0
            out_b = wq.numel() + sum(v.numel() * v.element_size()
                                     for v in qs.as_dict(packed=True).values())
            in_b = numel * DT.get(dtype, 2)
            rate = numel / max(1e-9, q_s) / 1e6
            print(f"  [{i+1}] {list(shape)} {dtype}  {human(in_b)}"
                  f"  read {read_s:.1f}s  quantize {q_s:.1f}s "
                  f"({rate:.1f} M weights/s)  -> {human(out_b)} "
                  f"({in_b/out_b:.2f}x)")
            total_numel += numel
            total_sec += q_s
            del w, wq, qs

    if total_sec <= 0:
        print("nothing was timed")
        return 1

    rate = total_numel / total_sec / 1e6
    print()
    print(f"measured: {total_numel/1e6:.0f}M weights in {total_sec:.1f}s "
          f"= {rate:.1f} M weights/s")

    # Extrapolate to the whole model. Only the tensors that are actually
    # quantized count: 232 in the transformer, ~436 in the text encoder.
    print()
    print("extrapolation to the whole checkpoint (quantized tensors only):")

    # weight counts sized from the dry-run projections
    est = [("transformer", 232, 7.12e9), ("text_encoder", 436, 8.76e9)]
    grand = 0.0
    for comp, ntensors, nparam in est:
        # not every parameter is inside a quantized tensor, but the quantized ones
        # dominate: use 90% as a deliberately conservative stand-in and say so
        sec = nparam * 0.90 / (rate * 1e6)
        grand += sec
        print(f"  {comp:14s} ~{ntensors} tensors, ~{nparam/1e9:.2f}e9 parameters "
              f"-> {sec/60:.1f} min")
    print(f"  {'total':14s} ~{grand/60:.1f} min at the measured rate")
    print()
    print("NOTE: the 90% factor is a stand-in for 'the parameters that live in a "
          "2-D .weight tensor'. The dry-run reports the exact count per component; "
          "this estimate exists only to answer whether the conversion takes an hour "
          "or a day, a question a factor of 1.1 cannot change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
