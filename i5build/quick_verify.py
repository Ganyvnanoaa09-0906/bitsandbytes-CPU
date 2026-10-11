"""quick_verify.py -- sample a complete .part against the mirror to decide reuse.

Context: a completed 9.28 GB shard was thrown away by a bug in the downloader (it
printed "re-verifying rather than assuming" and then truncated the file anyway). The
question now is whether an existing full-length file can be reused without paying a
full re-download.

The full check costs one ranged request per 4 MB window, which for a 9.28 GB file is
2377 requests and roughly as long as downloading it again. But the corruption this
project has actually seen was not random: verify_remote.py found whole 1 MB windows of
zeros where the mirror has dense data, and every mismatch had the same shape. Zero-fill
of a large region is detected by ANY sample inside that region, so a few dozen windows
spread across the file catches it with high probability, whereas a random single-bit
flip would not be caught by sampling at all.

So this is a screening test, and it is labelled as one:
  * it can prove a file BAD (a mismatch is conclusive)
  * it cannot prove a file GOOD (a clean sample only means the sampled windows agree)

Use it to avoid an obviously wasted re-download, then let the full verification run
afterwards as the real gate. For a first check on a freshly written file, 24-32 windows
spread evenly is the default.

Usage:
    python quick_verify.py <repo> <relpath> [<relpath> ...] [--samples 32]
"""
import argparse
import json
import os
import sys
import urllib.request

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}
WIN = 1 << 20


def remote_bytes(url, start, end, tries=3):
    h = dict(UA)
    h["Range"] = f"bytes={start}-{end}"
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                        timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"status {r.status}")
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("outdir")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--samples", type=int, default=32)
    a = ap.parse_args()

    api = f"{MIRROR}/api/models/{a.repo}/tree/main?recursive=true"
    with urllib.request.urlopen(urllib.request.Request(api, headers=dict(UA)),
                                timeout=60) as r:
        tree = json.loads(r.read().decode())
    sizes = {f["path"]: (f.get("size") or 0) for f in tree if f.get("type") == "file"}

    verdicts = {}
    for rel in a.paths:
        size = sizes.get(rel, 0)
        dest = os.path.join(a.outdir, rel.replace("/", os.sep))
        part = dest + ".part"
        src = part if os.path.isfile(part) else dest
        print(f"{rel}")
        if not size:
            print("  no advertised size; cannot judge")
            verdicts[rel] = None
            continue
        if not os.path.isfile(src):
            print("  absent")
            verdicts[rel] = None
            continue
        got = os.path.getsize(src)
        print(f"  {src}  {got:,} / {size:,} B")
        if got != size:
            print(f"  INCOMPLETE: {size-got:,} B short; reuse is not possible")
            verdicts[rel] = False
            continue
        url = f"{MIRROR}/{a.repo}/resolve/main/{rel}"
        n = a.samples
        mism = []
        for i in range(n):
            off = (got - WIN) * i // max(1, n - 1)
            with open(src, "rb") as fh:
                fh.seek(off)
                loc = fh.read(WIN)
            try:
                rem = remote_bytes(url, off, off + len(loc) - 1)
            except Exception as e:  # noqa: BLE001
                print(f"    sample {i} at {off:,}: fetch failed {e}")
                mism.append((off, "fetch-error"))
                continue
            if rem != loc:
                lz = loc.count(0) == len(loc)
                rz = rem.count(0) == len(rem)
                mism.append((off, f"differs (local all_zero={lz}, remote all_zero={rz})"))
        if mism:
            print(f"  BAD: {len(mism)} of {n} sampled windows differ")
            for off, why in mism[:6]:
                print(f"    {off:,}: {why}")
            verdicts[rel] = False
        else:
            print(f"  all {n} sampled windows match. NOT a proof of integrity -- a")
            print(f"  clean sample means the sampled windows agree. The full")
            print(f"  verification still has to run before this file is trusted.")
            verdicts[rel] = True
        print()

    print("verdicts:")
    for rel, v in verdicts.items():
        print(f"  {'usable (sampled clean)' if v else 'BAD' if v is False else 'unknown'}"
              f"  {rel}")
    return 0 if all(v is not False for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
