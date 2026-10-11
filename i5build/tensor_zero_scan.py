"""tensor_zero_scan.py -- which tensors are zero, and is that plausible?

shard_integrity.py found the transformer shard is 70.4% zero BYTES while its data
region matches the header exactly, so the file is not truncated. The VAE, a known
good control, is 0.3% zero. Two readings again:

  A. the file's contents were replaced with zeros while preserving length
  B. the zeros are in the tensors themselves

Note that a zero BYTE is not the same as a zero VALUE: in bf16 a zero value is two
zero bytes, and in fp32 it is four, so a small number of entirely-zero fp32 tensors
can account for a large fraction of the bytes. The per-tensor breakdown is what
separates A from B, and it needs no heuristic: a trained linear weight is never
identically zero, so counting exactly-zero tensors by name is decisive.

Reads every tensor from every transformer shard and reports, per shard: how many
tensors are all-zero, how many bytes they account for, and the largest all-zero
tensor. A structural file is read sequentially, so this is I/O bound.

Usage:
    python tensor_zero_scan.py [<model dir>]
"""
import json
import os
import sys

import numpy as np

DT = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16, "I64": np.int64,
      "I32": np.int32, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_, "F64": np.float64}


def read_header(path):
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode()), 8 + n


def scan(path, limit_report=12):
    meta, data_start = read_header(path)
    meta.pop("__metadata__", None)
    total_t = total_b = zero_t = zero_b = 0
    zeros = []
    with open(path, "rb") as fh:
        for key, info in meta.items():
            if "data_offsets" not in info:
                continue
            s, e = info["data_offsets"]
            fh.seek(data_start + s)
            buf = fh.read(e - s)
            dt = DT[info["dtype"]]
            arr = np.frombuffer(buf, dtype=dt)
            nbytes = e - s
            total_t += 1
            total_b += nbytes
            if arr.size and not arr.any():
                zero_t += 1
                zero_b += nbytes
                n = 1
                for d in info["shape"]:
                    n *= d
                zeros.append((nbytes, key, info["dtype"], tuple(info["shape"])))
    return total_t, total_b, zero_t, zero_b, zeros


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\work\qwen_env\Qwen-Image-2.1"
    for comp in ("transformer", "text_encoder", "vae"):
        d = os.path.join(root, comp)
        if not os.path.isdir(d):
            continue
        print("=" * 78)
        print(f"{comp}")
        print("=" * 78)
        for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
            path = os.path.join(d, fn)
            tt, tb, zt, zb, zeros = scan(path)
            print(f"{fn}")
            print(f"  {tt} tensors, {tb/1e9:.2f} GB of data")
            print(f"  ALL-ZERO tensors: {zt} ({100*zt/max(1,tt):.1f}%), "
                  f"{zb/1e9:.2f} GB ({100*zb/max(1,tb):.1f}% of the data)")
            zeros.sort(reverse=True)
            for nb, key, dt, shape in zeros[:5]:
                print(f"    {nb/1e6:9.2f} MB  {dt:5s} {shape}  {key}")
            if len(zeros) > 5:
                print(f"    ... and {len(zeros)-5} more")
            print()
    print("reading:")
    print("  a trained linear or embedding weight has no reason to be identically")
    print("  zero, so any all-zero tensor of that kind is a defect in the file, not")
    print("  a property of the model. norm scales initialized to zero DO exist in")
    print("  some architectures, and those are small; the question is whether the")
    print("  all-zero set is confined to them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
