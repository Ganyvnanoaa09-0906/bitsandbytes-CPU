"""bw_probe.py -- measure aggregate throughput with N parallel range requests.

WHY THIS EXISTS
---------------
A single hf-mirror download sustains about 2.2 MB/s. The question that decides
whether the 31.76 GB fetch takes ~4 hours or ~1 hour is whether that ceiling
belongs to one TCP connection or to the whole network path:

  * per-connection ceiling -> N parallel connections give N x 2.2 MB/s
  * path ceiling           -> N parallel connections give 2.2 MB/s in total

The earlier attempt at answering this was worthless: it opened three connections
to three *small* files and each finished after a few hundred KB, so what it
actually measured was connect latency, not throughput. This probe avoids that by
downloading a fixed 8 MB Range from a multi-GB file, so every connection stays
busy for the whole measurement, and by reading until the byte count is reached
rather than until the server closes.

It reads from the *end* of the file, so it never touches the byte range a
concurrent download of the same file is still writing.

Usage:
    python bw_probe.py [--n 8] [--mb 8] [--repo Qwen/Qwen-Image-2.1]
                       [--path text_encoder/model-00001-of-00004.safetensors]
"""
import argparse
import json
import os
import threading
import time
import urllib.request

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}


def remote_size(url):
    req = urllib.request.Request(url, headers=dict(UA), method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def result_of(api, path):
    with urllib.request.urlopen(urllib.request.Request(api, headers=dict(UA)), timeout=60) as r:
        for f in json.loads(r.read().decode()):
            if f.get("path") == path and f.get("type") == "file":
                return f.get("size") or 0
    return 0


def worker(url, start, end, out, idx):
    """Download [start, end] and record elapsed and byte count under out[idx]."""
    headers = dict(UA)
    headers["Range"] = f"bytes={start}-{end}"
    got = 0
    t0 = None
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            # 200 means the Range was ignored and the whole multi-GB file is about
            # to arrive; abort rather than poison the measurement with one
            # connection that is downloading 100x more than the others.
            if resp.status != 206:
                out[idx] = {"err": f"status {resp.status}, Range ignored", "bytes": 0, "sec": 0.0}
                return
            want = end - start + 1
            t0 = time.time()
            while got < want:
                chunk = resp.read(min(1 << 20, want - got))
                if not chunk:
                    break
                got += len(chunk)
    except Exception as e:  # noqa: BLE001
        out[idx] = {"err": f"{type(e).__name__}: {str(e)[:80]}", "bytes": got,
                    "sec": (time.time() - t0) if t0 else 0.0}
        return
    out[idx] = {"err": None, "bytes": got, "sec": time.time() - t0}


def run(n, mb, url, size):
    span = mb << 20
    out = [None] * n
    threads = []
    # Every worker takes a distinct span at the tail of the file.
    for i in range(n):
        end = size - 1 - i * span
        start = end - span + 1
        if start < 0:
            raise SystemExit(f"file too small for {n} x {mb} MB at the tail")
        threads.append(threading.Thread(target=worker, args=(url, start, end, out, i)))
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    ok = [r for r in out if r and not r["err"]]
    bad = [r for r in out if r and r["err"]]
    total = sum(r["bytes"] for r in ok)
    # Sum of per-connection rates is not the aggregate; aggregate is total bytes
    # over wall time, which is what actually decides the download duration.
    agg = total / wall / 1e6
    best = max((r["bytes"] / r["sec"] / 1e6 for r in ok if r["sec"] > 0), default=0)
    print(f"n={n}  span={mb} MB each  wall={wall:.1f}s  ok={len(ok)}  failed={len(bad)}")
    print(f"  aggregate : {agg:.2f} MB/s   ({total/1e6:.0f} MB in {wall:.1f}s)")
    print(f"  best single connection in this run: {best:.2f} MB/s")
    for i, r in enumerate(out):
        if r is None:
            print(f"    [{i}] no result")
        elif r["err"]:
            print(f"    [{i}] {r['bytes']/1e6:7.1f} MB  {r['sec']:5.1f}s  ERR {r['err']}")
        else:
            print(f"    [{i}] {r['bytes']/1e6:7.1f} MB  {r['sec']:5.1f}s  "
                  f"{r['bytes']/max(r['sec'],1e-9)/1e6:5.2f} MB/s")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--repo", default="Qwen/Qwen-Image-2.1")
    ap.add_argument("--path", default="text_encoder/model-00001-of-00004.safetensors")
    ap.add_argument("--api", action="store_true",
                    help="get size from the API instead of HEAD (mirror may omit Content-Length)")
    a = ap.parse_args()

    url = f"{MIRROR}/{a.repo}/resolve/main/{a.path}"
    if a.api:
        api = f"{MIRROR}/api/models/{a.repo}/tree/main?recursive=true"
        size = result_of(api, a.path)
    else:
        size = remote_size(url)
    print(f"url  : {url}")
    print(f"size : {size:,} B ({size/1e9:.2f} GB)")
    print()
    return 0 if run(a.n, a.mb, url, size) else 0


if __name__ == "__main__":
    raise SystemExit(main())
