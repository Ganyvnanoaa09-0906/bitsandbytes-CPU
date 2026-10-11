"""state_check.py -- what is actually on disk right now, with md5s.

Written after a self-inflicted incident: a first version of the segmented fetcher
accepted an HTTP 200 where it demanded 206, wrote nothing for every segment, then
truncated the destination to the advertised length. The result was a file of the
correct size filled with zeros, which the size check happily called complete. Two
small files in the destination tree were overwritten that way before it was caught.

So this reports size AND md5 for every candidate copy of the affected files, and
compares the destination tree against the ModelScope listing, which was fetched
independently and is a different host.

Usage:
    python state_check.py
"""
import hashlib
import json
import os
import urllib.request

DEST = r"D:\work\qwen_env\Qwen-Image-2.1"
UA = {"User-Agent": "Mozilla/5.0"}
MS_API = ("https://modelscope.cn/api/v1/models/Qwen/Qwen-Image-2.1/repo/files"
          "?Revision=master&Recursive=true")
HF_API = "https://hf-mirror.com/api/models/Qwen/Qwen-Image-2.1/tree/main?recursive=true"

SUSPECT = [
    "processor/merges.txt",
    "text_encoder/config.json",
    "processor/tokenizer.json",
]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def zeros_fraction(path, limit=4 << 20):
    """How much of the first `limit` bytes is zero -- a truncate-fill artefact shows
    up here immediately, where a size check cannot see it."""
    with open(path, "rb") as fh:
        data = fh.read(limit)
    if not data:
        return 1.0
    return data.count(0) / len(data)


def api_sizes():
    out = {}
    with urllib.request.urlopen(urllib.request.Request(HF_API, headers=UA), timeout=60) as r:
        for f in json.loads(r.read().decode()):
            if f.get("type") == "file":
                out[f["path"]] = f.get("size") or 0
    ms = {}
    with urllib.request.urlopen(urllib.request.Request(MS_API, headers=UA), timeout=60) as r:
        for f in json.loads(r.read().decode())["Data"]["Files"]:
            if f.get("Type") == "blob":
                ms[f["Path"]] = f.get("Size") or 0
    return out, ms


def main():
    hf, ms = api_sizes()
    print(f"{len(hf)} files on hf-mirror, {len(ms)} on modelscope")
    print()

    print("=== files where the two hosts disagree on size (suspicious) ===")
    for p in sorted(set(hf) & set(ms)):
        if hf[p] != ms[p]:
            print(f"  {p}\n      hf={hf[p]:,}  ms={ms[p]:,}  diff={hf[p]-ms[p]:+,}")
    print()

    print("=== destination tree vs advertised sizes ===")
    total = 0
    for p in sorted(hf):
        local = os.path.join(DEST, p.replace("/", os.sep))
        if not os.path.isfile(local):
            continue
        got = os.path.getsize(local)
        total += got
        want = hf[p]
        flag = "OK" if got == want else "PARTIAL" if got < want else "TOO BIG"
        extra = ""
        if p in SUSPECT or got == want:
            z = zeros_fraction(local)
            if z > 0.5:
                extra = f"   ZERO-FILLED {z*100:.1f}%"
                flag = "CORRUPT"
        print(f"  {flag:9s} {got:>13,} / {want:>13,}  {p}{extra}")
    print(f"\n  tree total: {total/1e9:.2f} GB")
    print()

    print("=== md5 of the files that are suspect ===")
    for p in SUSPECT:
        local = os.path.join(DEST, p.replace("/", os.sep))
        if os.path.isfile(local):
            print(f"  {p}: {os.path.getsize(local):,}  {md5(local)}")
        else:
            print(f"  {p}: absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
