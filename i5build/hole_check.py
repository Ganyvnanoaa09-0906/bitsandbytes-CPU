"""hole_check.py -- is the corruption sparse holes from the writer, not bad bytes?

The evidence so far: remote bytes are correct wherever the local file is non-zero,
and every local zero region is non-zero remotely. Controls match byte for byte at
three different offsets across two hosts, so the comparison and the network are both
sound, and the local copy is what is wrong.

The mechanism that fits is the segmented writer's resume logic. It used
`os.path.getsize(dest)` as "how many bytes have arrived" and then, for each segment,
did `fh.seek(pos)` before writing. File LENGTH and WRITTEN DATA are different things:
a segment writes at an absolute offset, so if the earlier run died between extending
the file and filling a region, the region is a SPARSE HOLE, and the file system
reports the full length while reads at those offsets return zeros. On two runs the
log recorded "getaddrinfo failed" for all segments at once, which is exactly the
interruption that leaves holes behind. The resume then trusted the length, seeked
past the holes, and never filled them.

So the zeros should not be a property of any tensor but a property of specific byte
RANGES, and those ranges should be block-aligned and cross tensor boundaries. This
tests that prediction against the alternative (whole tensors were overwritten with
zeros): it walks the file and maps the zero runs, then checks whether the run
boundaries land on allocation granularity rather than on tensor boundaries.

Also does the positive test: a non-zero region immediately before and after a zero
run should still match the remote, proving the file is correct everywhere except
inside the runs.

Usage:
    python hole_check.py
"""
import json
import os
import struct
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}
HF = "https://hf-mirror.com/Qwen/Qwen-Image-2.1/resolve/main/{path}"
LOCAL = r"D:\work\qwen_env\Qwen-Image-2.1"
REL = "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode()), 8 + n


def remote_slice(rel, start, end):
    h = dict(UA)
    h["Range"] = f"bytes={start}-{end}"
    with urllib.request.urlopen(
            urllib.request.Request(HF.format(path=rel), headers=h), timeout=90) as r:
        if r.status != 206:
            raise RuntimeError(f"status {r.status}")
        return r.read()


def zero_runs(path, start, limit=40):
    """Locate runs of zero bytes, scanning in 1 MB windows."""
    runs = []
    win = 1 << 20
    fsize = os.path.getsize(path)
    with open(path, "rb") as fh:
        pos = start
        run_start = None
        while pos < fsize and len(runs) < limit:
            fh.seek(pos)
            buf = fh.read(min(win, fsize - pos))
            if not buf:
                break
            # find zero runs inside this window
            i = 0
            n = len(buf)
            while i < n:
                if buf[i] == 0:
                    j = i
                    while j < n and buf[j] == 0:
                        j += 1
                    abs_start = pos + i
                    if run_start is None:
                        run_start = abs_start
                    i = j
                else:
                    if run_start is not None:
                        runs.append((run_start, pos + i))
                        run_start = None
                    i += 1
            if run_start is None:
                pass
            pos += len(buf)
        if run_start is not None:
            runs.append((run_start, fsize))
    return runs


def main():
    path = os.path.join(LOCAL, REL.replace("/", os.sep))
    meta, dstart = read_header(path)
    fsize = os.path.getsize(path)
    print(f"file      : {path}")
    print(f"size      : {fsize:,} B")
    print(f"data start: {dstart:,} B")
    print()

    print("mapping zero runs (a run that spans tensor boundaries is a hole; a run")
    print("that stops exactly at tensor boundaries would be a zeroed tensor)")
    runs = zero_runs(path, dstart)
    print(f"  found {len(runs)} run(s) in the first pass")
    for a, b in runs[:12]:
        print(f"    {a:15,d} .. {b:15,d}   ({b-a:12,d} B)")
    print()

    # Do the run boundaries coincide with tensor boundaries?
    bounds = set()
    for k, v in meta.items():
        if "data_offsets" in v:
            bounds.add(dstart + v["data_offsets"][0])
            bounds.add(dstart + v["data_offsets"][1])
    if runs:
        a, b = runs[0]
        print(f"first run start {a:,}: on a tensor boundary? {a in bounds}")
        print(f"first run end   {b:,}: on a tensor boundary? {b in bounds}")
        size = b - a
        print(f"run length {size:,} B = {size/1024:.1f} KB = {size/1048576:.3f} MB")
        print(f"  divisible by 4096? {size % 4096 == 0}")
        print(f"  divisible by 65536? {size % 65536 == 0}")
    print()

    # Positive test: bytes just outside a hole must match the remote.
    print("positive test: compare remote bytes immediately around the first run")
    if runs:
        a, b = runs[0]
        for label, s in (("just before", max(dstart, a - 4096)),
                         ("just after", min(fsize - 4096, b))):
            e = s + 4096 - 1
            with open(path, "rb") as fh:
                fh.seek(s)
                loc = fh.read(4096)
            try:
                rem = remote_slice(REL, s, e)
                same = rem == loc
                print(f"  {label:12s} offset {s:15,d}: "
                      f"{'SAME as remote' if same else 'DIFFERS from remote'} "
                      f"(local zeros {loc.count(0)}/4096)")
            except Exception as ex:  # noqa: BLE001
                print(f"  {label:12s} offset {s:15,d}: remote error {ex}")
    print()
    print("reading:")
    print("  runs that span tensor boundaries and are block-aligned are holes left")
    print("  by seeking past unwritten regions, i.e. a resume bug. Runs that align")
    print("  exactly with tensor boundaries would instead mean whole tensors were")
    print("  zeroed, i.e. corrupt data. The two need different fixes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
