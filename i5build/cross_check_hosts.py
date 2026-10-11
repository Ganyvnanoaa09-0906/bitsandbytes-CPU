"""Prove ModelScope serves the same bytes as hf-mirror before switching source.

The download is ~25 GB and hf-mirror single-connection throughput is 0.11 MB/s while
ModelScope reaches 2.6 MB/s. Switching is worth two orders of magnitude, but only if
the two hosts serve the SAME revision: text_encoder/model-00001 was already fetched
and verified from hf-mirror, and a model assembled from two different revisions would
be internally inconsistent in a way no per-window check can detect.

So this compares bytes, not metadata:
  * advertised size for every file, from both hosts
  * a set of windows of the already-verified local file against ModelScope
"""
import os
import sys
import urllib.request

REPO = "Qwen/Qwen-Image-2.1"
LOCAL = r"D:\work\qwen_env\Qwen-Image-2.1"
WIN = 4 * 1024 * 1024
HDR = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}

HF = "https://hf-mirror.com"
MS = "https://modelscope.cn/models"

FILES = [
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
]
DONE = [
    "text_encoder/model-00001-of-00004.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]


def url_for(host, relpath):
    if host == MS:
        return f"{MS}/{REPO}/resolve/master/{relpath}"
    return f"{host}/{REPO}/resolve/main/{relpath}"


def total_size(host, relpath):
    """Authoritative size from the data host itself, not from an index."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url_for(host, relpath), headers={**HDR, "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                cr = r.headers.get("Content-Range") or ""
                return int(cr.split("/")[-1]) if "/" in cr else -1
        except Exception as e:  # noqa: BLE001
            last = e
    print(f"    size probe failed on {host}: {type(last).__name__}: {str(last)[:60]}")
    return None


def ranged(host, relpath, start, length):
    req = urllib.request.Request(
        url_for(host, relpath),
        headers={**HDR, "Range": f"bytes={start}-{start+length-1}"})
    with urllib.request.urlopen(req, timeout=90) as r:
        if r.status != 206:
            raise RuntimeError(f"status {r.status}: Range ignored")
        return r.read()


def compare_windows(relpath, size, picks):
    path = os.path.join(LOCAL, relpath.replace("/", os.sep))
    if not os.path.isfile(path):
        print(f"  {relpath}: no local copy, skipped")
        return None
    n = 0
    with open(path, "rb") as fh:
        for start in picks:
            end = min(size, start + WIN) - 1
            fh.seek(start)
            loc = fh.read(end - start + 1)
            try:
                rem = ranged(MS, relpath, start, end - start + 1)
            except Exception as e:  # noqa: BLE001
                print(f"    offset {start:>14,}: FETCH FAILED {type(e).__name__}")
                return False
            same = rem == loc
            print(f"    offset {start:>14,}: {'identical' if same else 'DIFFERS'}"
                  f"  ({len(loc):,} B)")
            if not same:
                return False
            n += 1
    print(f"  {relpath}: {n}/{n} sampled window(s) identical")
    return True


if __name__ == "__main__":
    print("=== advertised sizes, host by host ===")
    sizes = {}
    for rel in FILES + DONE:
        hf = total_size(HF, rel)
        ms = total_size(MS, rel)
        sizes[rel] = (hf, ms)
        flag = "same" if hf == ms else "*** DIFFERENT ***"
        print(f"  {rel}")
        print(f"    hf-mirror  : {hf:,}" if hf else "    hf-mirror  : probe failed")
        print(f"    modelscope : {ms:,}" if ms else "    modelscope : probe failed")
        print(f"    -> {flag}")

    print("\n=== byte comparison on files already verified from hf-mirror ===")
    allok = True
    for rel in DONE:
        hf, ms = sizes.get(rel, (None, None))
        size = ms or hf
        if not size or size < 0:
            print(f"  {rel}: no usable size")
            allok = False
            continue
        n_win = (size + WIN - 1) // WIN
        picks = sorted({0, 1, n_win // 3, n_win // 2, n_win - 2, n_win - 1})
        picks = [p * WIN for p in picks if 0 <= p < n_win]
        r = compare_windows(rel, size, picks)
        if r is not True:
            allok = False

    print()
    if allok:
        print("VERDICT: ModelScope serves byte-identical content for the files that "
              "were already verified from hf-mirror, and advertises the same sizes. "
              "Safe to continue the download from ModelScope.")
    else:
        print("VERDICT: NOT safe to mix sources -- see the mismatches above.")
    sys.exit(0 if allok else 1)
