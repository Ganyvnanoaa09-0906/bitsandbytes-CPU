"""Measure HF CDN throughput with 1 vs N parallel connections.

Decides whether the current single-threaded re-verify pass is the bottleneck,
or whether the link itself is saturated.
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "Qwen/Qwen-Image-2.1"
FILE = "text_encoder/model-00002-of-00004.safetensors"
BASE = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
URL = f"{BASE}/{REPO}/resolve/main/{FILE}"

TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
HEADERS = {"User-Agent": "speed-probe/1.0"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

WIN = 4 * 1024 * 1024


def fetch_range(start, length):
    req = urllib.request.Request(URL, headers={**HEADERS, "Range": f"bytes={start}-{start + length - 1}"})
    t0 = time.time()
    n = 0
    with urllib.request.urlopen(req, timeout=60) as r:
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            n += len(chunk)
    return n, time.time() - t0


def run(nconn, total_mb):
    total = total_mb * 1024 * 1024
    t0 = time.time()
    if nconn == 1:
        got, _ = fetch_range(0, total)
    else:
        per = max(WIN, total // nconn)
        per -= per % WIN
        with ThreadPoolExecutor(max_workers=nconn) as ex:
            futs = [ex.submit(fetch_range, i * per, per) for i in range(nconn)]
            got = sum(f.result()[0] for f in futs)
    dt = time.time() - t0
    print(f"  conns={nconn:2d}  {got/1048576:7.2f} MB in {dt:6.2f} s  -> {got/1048576/dt:6.3f} MB/s")
    return got / 1048576 / dt


if __name__ == "__main__":
    print(f"URL: {URL}")
    print(f"token: {'yes' if TOKEN else 'no'}")
    print("warmup / reachability ...")
    try:
        n, dt = fetch_range(0, 1024 * 1024)
        print(f"  1 MB in {dt:.2f}s ({n/1048576/dt:.3f} MB/s)")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print("\nthroughput by connection count:")
    rates = {}
    for k in (1, 2, 4, 8, 12):
        try:
            rates[k] = run(k, 8)
        except Exception as exc:
            print(f"  conns={k:2d}  FAILED: {type(exc).__name__}: {exc}")
            rates[k] = 0.0

    best = max(rates, key=lambda k: rates[k])
    print(f"\nbest: {best} conns at {rates[best]:.3f} MB/s  (1 conn = {rates.get(1, 0):.3f} MB/s)")
    if rates.get(1, 0) > 0 and rates[best] / max(rates[1], 1e-9) > 1.5:
        print("=> parallelism helps a lot: the single-threaded verify pass is the bottleneck")
    else:
        print("=> parallelism does not help much: the link itself is saturated")
