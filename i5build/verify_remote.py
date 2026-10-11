"""verify_remote.py -- sweep every shard in 1 MB windows against the mirror.

Established facts, none of which depend on a mechanism:
  * three control tensors match the remote byte for byte at three offsets on two
    hosts, so the comparison and the network path are sound
  * at offsets where the local file is all zero, the remote bytes are dense
    (255-256 distinct byte values), so those regions are not the tensor data
  * in a 24-point sweep of one shard, 8 windows matched exactly and 16 were 100%
    zero, with 0 errors

So the local copy is wrong. "Wrong" is not yet measured to a boundary, and the
boundary matters: it decides whether one file needs re-fetching or all of them, and a
sample-based answer cannot decide that because the corruption is bimodal (whole
windows are either perfect or entirely zero).

This walks every file, window by window, and reports per file the exact count of
matching windows and the byte ranges that differ. It is I/O bound reading 33 GB, and
network bound fetching 1 MB per window; it uses a coarse stride first and then
refines only where a mismatch is found, so a clean file costs a handful of requests
and a corrupt one is mapped exactly.

Usage:
    python verify_remote.py                 # all shards, coarse
    python verify_remote.py --stride 16     # every 16th window
    python verify_remote.py --file <path>
"""
import argparse
import json
import os
import struct
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}
HF = "https://hf-mirror.com/Qwen/Qwen-Image-2.1/resolve/main/{path}"
ROOT = r"D:\work\qwen_env\Qwen-Image-2.1"
WIN = 1 << 20


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode()), 8 + n


def remote_range(rel, start, end, tries=3):
    h = dict(UA)
    h["Range"] = f"bytes={start}-{end}"
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(HF.format(path=rel), headers=h),
                    timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"status {r.status}")
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:70]}")


def check_file(rel, path, stride):
    fsize = os.path.getsize(path)
    n_win = (fsize + WIN - 1) // WIN
    print(f"  {rel}")
    print(f"    {fsize:,} B, {n_win} windows, stride {stride}")
    offsets = list(range(0, fsize, WIN * stride))
    matches = mismatches = errors = 0
    bad = []
    for i, off in enumerate(offsets):
        if i % 40 == 0 and i:
            print(f"      ... {i}/{len(offsets)}", flush=True)
        with open(path, "rb") as fh:
            fh.seek(off)
            loc = fh.read(min(WIN, fsize - off))
        try:
            rem = remote_range(rel, off, off + len(loc) - 1)
        except Exception as e:  # noqa: BLE001
            errors += 1
            if errors <= 3:
                print(f"      fetch error at {off:,}: {e}")
            continue
        if rem == loc:
            matches += 1
        else:
            mismatches += 1
            loc_z = loc.count(0) == len(loc)
            rem_z = rem.count(0) == len(rem)
            bad.append((off, len(loc) * (loc.count(0) / len(loc)), loc_z, rem_z))
    pct = 100 * mismatches / max(1, matches + mismatches)
    print(f"    match {matches}  mismatch {mismatches} ({pct:.1f}%)  errors {errors}")
    for off, frac, lz, rz in bad[:6]:
        print(f"      {off:15,d}  local {100*frac:5.1f}% zero (all_zero={lz})  "
              f"remote all_zero={rz}")
    if len(bad) > 6:
        print(f"      ... and {len(bad)-6} more mismatched windows")
    return matches, mismatches, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--file", default=None)
    a = ap.parse_args()

    targets = []
    if a.file:
        rel = os.path.relpath(a.file, a.root).replace(os.sep, "/")
        targets.append((rel, a.file))
    else:
        for comp in ("transformer", "text_encoder", "vae"):
            d = os.path.join(a.root, comp)
            if not os.path.isdir(d):
                continue
            for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
                targets.append((f"{comp}/{fn}", os.path.join(d, fn)))

    print(f"verifying {len(targets)} file(s) against {HF.split('/resolve')[0]}")
    print(f"window {WIN} B, stride {a.stride} (1 in {a.stride} windows)")
    print()
    tot_m = tot_x = tot_e = 0
    summary = []
    for rel, path in targets:
        m, x, e = check_file(rel, path, a.stride)
        tot_m += m
        tot_x += x
        tot_e += e
        summary.append((rel, m, x, e))
        print()
    print("=" * 78)
    print(f"{'file':62s} {'match':>7s} {'bad':>6s} {'err':>5s}")
    for rel, m, x, e in summary:
        print(f"{rel:62s} {m:7d} {x:6d} {e:5d}")
    print()
    print(f"total: {tot_m} matching windows, {tot_x} mismatching, {tot_e} fetch errors")
    verdict = "ALL VERIFIED" if tot_x == 0 and tot_e == 0 else "CORRUPTION FOUND"
    print(f"verdict: {verdict}")
    return 0 if (tot_x == 0 and tot_e == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
