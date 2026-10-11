"""safetensors_head.py -- read tensor names/shapes from a partially downloaded shard.

A safetensors file is a u64 header length, then that much JSON, then the raw
tensor data. The header is at the very start, so a shard that is still 3% of the
way through downloading already describes every tensor it will contain. That makes
it possible to settle the quantization plan -- which keys get quantized, which keep
bf16, what the module classes are, what the resulting size will be -- while the
download is still running, instead of discovering at the end that the rule was
wrong for this architecture.

Reports, per shard:
  * tensor count, dtype mix, and total bytes
  * the class-name histogram taken from the key paths
  * every 2-D tensor ending in .weight (the quantization candidates) with its
    shape, so a pattern of "only q_proj/k_proj/..." can be confirmed or refuted
  * what a 4-bit conversion would save, and what it would leave in bf16

Usage:
    python safetensors_head.py <file> [<file> ...]
"""
import collections
import json
import os
import struct
import sys


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def read_header(path):
    """(metadata_dict, header_bytes) without reading the tensor data."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise SystemExit(f"{path}: too short to hold a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 200 << 20:
            raise SystemExit(f"{path}: implausible header length {n} "
                             f"(zero-filled or corrupt)")
        blob = fh.read(n)
    return json.loads(blob.decode()), n


DTYPE_SIZE = {
    "F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2,
    "I16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    for path in sys.argv[1:]:
        size = os.path.getsize(path)
        print("=" * 78)
        print(f"{path}")
        print(f"  on disk now: {human(size)}")
        meta, hlen = read_header(path)
        meta.pop("__metadata__", None)
        print(f"  header: {human(hlen)}, {len(meta)} tensors")
        print()

        dtypes = collections.Counter()
        bytes_by_dtype = collections.Counter()
        total = 0
        for name, info in meta.items():
            dt = info["dtype"]
            n = 1
            for d in info["shape"]:
                n *= d
            b = n * DTYPE_SIZE.get(dt, 0)
            dtypes[dt] += 1
            bytes_by_dtype[dt] += b
            total += b
        print("  dtype mix:")
        for dt, c in dtypes.most_common():
            print(f"    {dt:8s} {c:5d} tensors  {human(bytes_by_dtype[dt])}")
        print(f"    {'total':8s} {len(meta):5d} tensors  {human(total)}")
        print()

        # The class-name histogram: the segment before the first digit-bearing
        # block tells us what kind of module each parameter belongs to.
        classes = collections.Counter()
        for name in meta:
            parts = [p for p in name.split(".") if not p.isdigit()]
            classes[".".join(parts[:2]) if len(parts) >= 2 else parts[0]] += 1
        print("  module paths (top 15):")
        for c, n in classes.most_common(15):
            print(f"    {n:5d}  {c}")
        print()

        # Quantization candidates and the size they would take.
        cand, keep = [], []
        for name, info in meta.items():
            n = 1
            for d in info["shape"]:
                n *= d
            b = n * DTYPE_SIZE.get(info["dtype"], 0)
            (cand if (len(info["shape"]) == 2 and name.endswith(".weight")) else keep).append(
                (name, info["shape"], b))
        cand_b = sum(b for _, _, b in cand)
        keep_b = sum(b for _, _, b in keep)
        # 4 bits per weight plus an fp32 absmax per 64 weights, then the nested
        # absmax is itself compressed to 8 bits per 64, so ~4.5 bits per weight.
        q_b = int(cand_b / 2 * (4 + 32 / 64) / 8)
        print(f"  quantization candidates (2-D, .weight): {len(cand)} "
              f"-> {human(cand_b)} becomes ~{human(q_b)}")
        print(f"  kept in full precision                 : {len(keep)} "
              f"-> {human(keep_b)}")
        print(f"  projected total: {human(q_b + keep_b)} "
              f"(from {human(total)}, {total/max(1, q_b+keep_b):.2f}x smaller)")
        print()

        # Are there 2-D .weight tensors that pattern-matching would have missed?
        # Group by the trailing two components to show the naming convention.
        tails = collections.Counter()
        for name, shape, _ in cand:
            tails[".".join(name.split(".")[-3:-1])] += 1
        print("  candidate key tails:")
        for t, n in tails.most_common(20):
            print(f"    {n:5d}  ...{t}.weight")
        print()

        # Non-.weight 2-D tensors are the interesting near-misses.
        odd = [(n, s) for n, s in
               ((n, i["shape"]) for n, i in meta.items())
               if len(s) == 2 and not n.endswith(".weight")]
        if odd:
            print(f"  2-D tensors NOT ending in .weight ({len(odd)}):")
            for n, s in odd[:15]:
                print(f"    {list(s)}  {n}")
            print()
        print(f"  NOTE: the header describes the whole file, but only "
              f"{human(max(0, size - hlen - 8))} of the tensor data is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
