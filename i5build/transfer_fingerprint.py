"""transfer_fingerprint.py -- what did the transfer actually deliver?

The zero pattern is bimodal: 28.7% of 1 MB windows are 0-10% zero and 71.3% are
100% zero, with a 2,229 MB all-zero span. Real bf16 weights are nowhere near 0%
zero bytes, so "0-10%" cannot mean "intact" -- it means those windows are also not
the tensor data. That points at the transfer layer, not at any tensor.

Untested hypothesis: the response carried a content encoding (gzip, br, deflate) and
the downloader wrote the compressed stream while the offsets in the safetensors
header describe the decompressed layout. A compressed stream is high-entropy, which
would NOT be 71% zeros -- unless the two regions came from different responses, one
compressed and one not, which is exactly what a resume that switched endpoints
between attempts could produce.

So: read the response headers for the same URL the downloader used, then fingerprint
the local bytes inside a "0-10%" window to see whether they look like tensor data at
all. Comparing a window's bytes against the remote bytes for the same range settles
it without any theory.

Usage:
    python transfer_fingerprint.py
"""
import json
import os
import struct
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}
HF = "https://hf-mirror.com/Qwen/Qwen-Image-2.1/resolve/main/{path}"
LOCAL = r"D:\work\qwen_env\Qwen-Image-2.1"
REL = "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
WIN = 1 << 20


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode()), 8 + n


def classify(buf):
    """A crude shape test: real bf16 tensor data has few zero bytes and a wide byte
    histogram; text or a compressed stream look different again."""
    n = len(buf)
    zeros = buf.count(0)
    hist = [0] * 256
    for b in buf[:200000]:
        hist[b] += 1
    distinct = sum(1 for c in hist if c)
    top = sorted(hist, reverse=True)[:3]
    return (f"zeros {100*zeros/n:5.1f}%  distinct byte values {distinct}/256  "
            f"top3 {top[0]},{top[1]},{top[2]}")


def main():
    path = os.path.join(LOCAL, REL.replace("/", os.sep))
    fsize = os.path.getsize(path)

    print("=== response headers for the URL the downloader used ===")
    for label, rng in (("plain GET", None), ("ranged GET", "bytes=0-1048575")):
        h = dict(UA)
        if rng:
            h["Range"] = rng
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(HF.format(path=REL), headers=h),
                    timeout=60) as r:
                print(f"  {label}: status {r.status}")
                for k in ("Content-Type", "Content-Length", "Content-Encoding",
                          "Content-Range", "Transfer-Encoding", "Vary", "ETag"):
                    v = r.headers.get(k)
                    if v:
                        print(f"      {k}: {v}")
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: {type(e).__name__}: {str(e)[:90]}")
    print()

    # Two windows: one from the 0-10% bucket and one from the 100% bucket.
    meta, dstart = read_header(path)
    probes = []
    with open(path, "rb") as fh:
        pos = dstart
        while pos < fsize and len(probes) < 2:
            fh.seek(pos)
            buf = fh.read(WIN)
            if not buf:
                break
            frac = buf.count(0) / len(buf)
            if frac < 0.10 and not any(p[2] == "low" for p in probes):
                probes.append((pos, buf[:4096], "low"))
            if frac == 1.0 and not any(p[2] == "all" for p in probes):
                probes.append((pos, buf[:4096], "all"))
            pos += WIN
    print("=== fingerprints ===")
    for pos, head, kind in probes:
        print(f"  window at {pos:15,d} ({kind} zero): {classify(head)}")
    print()

    # Fetch the same ranges remotely and compare.
    print("=== the same ranges, fetched now ===")
    for pos, head, kind in probes:
        h = dict(UA)
        h["Range"] = f"bytes={pos}-{pos + 4095}"
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(HF.format(path=REL), headers=h),
                    timeout=90) as r:
                rem = r.read()
            same = rem == head
            print(f"  {pos:15,d} ({kind}): remote {classify(rem)}  "
                  f"-> {'SAME' if same else 'DIFFERS'}")
        except Exception as e:  # noqa: BLE001
            print(f"  {pos:15,d} ({kind}): remote error {type(e).__name__}")
    print()

    # Does the local file's data region equal the remote bytes anywhere at all?
    print("=== how much of the file agrees with the remote? (1 MB samples) ===")
    agree = disagree = errors = 0
    for i in range(24):
        off = dstart + (fsize - dstart - WIN) * i // 23
        with open(path, "rb") as fh:
            fh.seek(off)
            loc = fh.read(WIN)
        h = dict(UA)
        h["Range"] = f"bytes={off}-{off + len(loc) - 1}"
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(HF.format(path=REL), headers=h),
                    timeout=120) as r:
                rem = r.read()
            if rem == loc:
                agree += 1
            else:
                disagree += 1
                if disagree <= 3:
                    print(f"    differs at {off:15,d}: local {classify(loc[:4096])}")
                    print(f"    {'':>22} remote {classify(rem[:4096])}")
        except Exception:  # noqa: BLE001
            errors += 1
    print(f"  agree {agree}, differ {disagree}, error {errors} out of 24 samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
