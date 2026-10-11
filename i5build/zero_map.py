"""zero_map.py -- where are the zeros, at 1 MB resolution?

The previous attempts to explain the zeros produced two wrong mechanisms (a
"first-row reuse" bug that the A/B disproved, then sparse holes, which hole_check.py
disproved by finding the zero runs to be single bytes and the bytes around them to
match the remote). The reading itself is settled: remote bytes at offsets where the
local file is zero are NON-ZERO on two independent hosts, while three control
tensors match byte for byte. So the local file differs from the source over large
regions.

What is still unknown is the SHAPE of those regions, and the shape is what identifies
the cause: a whole-tensor zeroing, a contiguous unflushed tail, or scattered windows.
This maps zero density per 1 MB block and lists the largest contiguous zero spans, no
theory attached.

Usage:
    python zero_map.py [<file>]
"""
import os
import sys

LOCAL = r"D:\work\qwen_env\Qwen-Image-2.1"
DEFAULT = os.path.join(LOCAL, "transformer",
                       "diffusion_pytorch_model-00001-of-00002.safetensors")
WIN = 1 << 20


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    fsize = os.path.getsize(path)
    print(f"file: {path}")
    print(f"size: {fsize:,} B ({fsize/1e9:.2f} GB)")
    print()

    densities = []
    with open(path, "rb") as fh:
        pos = 0
        while pos < fsize:
            fh.seek(pos)
            buf = fh.read(min(WIN, fsize - pos))
            if not buf:
                break
            densities.append(buf.count(0) / len(buf))
            pos += len(buf)

    n = len(densities)
    print(f"1 MB windows: {n}")
    buckets = {"0%": 0, "0-10%": 0, "10-50%": 0, "50-90%": 0, "90-99%": 0,
               "99-100%": 0, "100% (all zero)": 0}
    for d in densities:
        if d == 0:
            buckets["0%"] += 1
        elif d < 0.10:
            buckets["0-10%"] += 1
        elif d < 0.50:
            buckets["10-50%"] += 1
        elif d < 0.90:
            buckets["50-90%"] += 1
        elif d < 0.99:
            buckets["90-99%"] += 1
        elif d < 1.0:
            buckets["99-100%"] += 1
        else:
            buckets["100% (all zero)"] += 1
    print("zero density per 1 MB window:")
    for k, v in buckets.items():
        print(f"  {k:16s} {v:6d}  ({100*v/max(1,n):5.1f}%)  "
              f"{'#' * int(60*v/max(1,n))}")
    print()

    # Largest contiguous run of all-zero windows, reported in MB.
    best_len = best_start = cur_len = 0
    cur_start = 0
    for i, d in enumerate(densities):
        if d == 1.0:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    print(f"largest contiguous all-zero span: {best_len} MB "
          f"starting at offset {best_start*WIN:,}")
    print()

    # Does the run structure look page aligned? A hole from a skipped write would be,
    # a zeroed tensor would not be.
    print("alignment of the all-zero windows: first and last index")
    idx = [i for i, d in enumerate(densities) if d == 1.0]
    if idx:
        print(f"  all-zero windows: {len(idx)} of {n}")
        print(f"  first at {idx[0]*WIN:,} B, last at {idx[-1]*WIN:,} B")
        print(f"  contiguous? {idx == list(range(idx[0], idx[-1] + 1))}")
    else:
        print("  none at 1 MB resolution, so the zeros are finer than 1 MB")
    print()
    print("compare with the tensor map:")
    print("  if the all-zero windows coincide with whole tensors, the content of")
    print("  those tensors is zero. If they straddle tensor boundaries, something")
    print("  wrote zeros over a byte range instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
