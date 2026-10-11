"""shard_integrity.py -- is the checkpoint itself corrupt, or is the model really zero?

weight_sanity.py found transformer_blocks.0.attn.to_q.weight to be identically zero
over 16.7M elements, read three independent ways (a hand-written raw parser, 
safetensors.load_file and get_slice), and the quantizer's output for it is a constant
uint8, which is what quantizing a constant produces. So the reading is not an
artefact of one library.

That leaves two very different explanations:

  A. the checkpoint on disk is corrupt -- truncated, or padded with zeros, in a way
     that passes a size check. This session has already proved that a size check
     cannot see a zero-filled file, and the model was downloaded over a link that
     dropped twice with "getaddrinfo failed" before resuming.
  B. the tensors really are zero in the published model, which would make the whole
     quantization exercise meaningless.

The test that separates them is the FILE, not any single tensor. A safetensors file
is a header of declared tensor offsets followed by the tensor data. If the data
region is short, or if the bytes outside a few tensors are largely zero, the file is
the problem. If the file is structurally sound and the zeros are confined to
specific tensors, B holds.

Checks, none of which require loading a tensor:
  * declared data size versus actual file size
  * the fraction of zero bytes in the whole data region, sampled
  * the same for each shard, and for the VAE as a control (it is known to load and
    was verified to have 238 tensors and a sane header)

Usage:
    python shard_integrity.py [<model dir>]
"""
import json
import os
import struct
import sys

DT = {"BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1,
      "BOOL": 1, "F64": 8}


def read_header(path):
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return None, 0, 0
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 500 << 20:
            return None, 8, n
        blob = fh.read(n)
    try:
        meta = json.loads(blob.decode())
    except Exception:
        return None, 8 + n, n
    return meta, 8 + n, n


def zero_fraction(path, start, end, samples=64):
    """Fraction of zero bytes in `samples` evenly spaced 1 MB windows."""
    size = end - start
    if size <= 0:
        return None, 0
    win = min(1 << 20, max(4096, size // samples))
    tot = zeros = 0
    with open(path, "rb") as fh:
        for i in range(samples):
            off = start + (size - win) * i // max(1, samples - 1)
            fh.seek(off)
            buf = fh.read(win)
            tot += len(buf)
            zeros += buf.count(0)
    return (zeros / tot if tot else None), tot


def check(path):
    name = os.path.basename(path)
    fsize = os.path.getsize(path)
    meta, data_start, hlen = read_header(path)
    print(f"--- {name}")
    print(f"    file size      : {fsize:,} B ({fsize/1e9:.2f} GB)")
    if meta is None:
        print(f"    HEADER UNREADABLE (declared header length {hlen:,})")
        return False
    meta.pop("__metadata__", None)
    declared_end = 0
    n_tensors = 0
    for k, v in meta.items():
        if "data_offsets" in v:
            declared_end = max(declared_end, v["data_offsets"][1])
            n_tensors += 1
    data_bytes = fsize - data_start
    print(f"    header         : {hlen:,} B, {n_tensors} tensors")
    print(f"    data region    : {data_bytes:,} B; header declares {declared_end:,} B")
    diff = data_bytes - declared_end
    ok_size = abs(diff) < 4096
    print(f"    difference     : {diff:+,} B  -> {'consistent' if ok_size else 'MISMATCH'}")
    zf, tot = zero_fraction(path, data_start, fsize)
    if zf is not None:
        print(f"    zero bytes     : {zf*100:5.1f}% of {tot/1e6:.1f} MB sampled")
    return ok_size


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\work\qwen_env\Qwen-Image-2.1"
    print(f"model: {root}")
    print()
    allok = True
    for comp in ("transformer", "text_encoder", "vae"):
        d = os.path.join(root, comp)
        if not os.path.isdir(d):
            continue
        print(f"=== {comp} ===")
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".safetensors"):
                allok &= check(os.path.join(d, fn))
        print()
    print("reading:")
    print("  a file whose data region is shorter than its header declares is")
    print("  truncated. A file that is size-consistent but mostly zero bytes is")
    print("  padded. Both mean the download produced a file that a size check")
    print("  would have called complete -- the same failure mode as the")
    print("  zero-filled segment downloader that started this line of work.")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
