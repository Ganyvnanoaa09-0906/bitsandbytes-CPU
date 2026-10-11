"""Compare download hosts and parallelism for the Qwen-Image-2.1 shards.

The fetch has been running for hours at a few hundred KB/s because the reuse/verify
paths in hf_safe_fetch.py are single-threaded. This measures whether the ceiling is
the link or the number of connections, and whether another host is simply faster.
"""
import os
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

REPO = "Qwen/Qwen-Image-2.1"
FILE = "text_encoder/model-00002-of-00004.safetensors"

HOSTS = {
    "hf-mirror":  f"https://hf-mirror.com/{REPO}/resolve/main/{FILE}",
    "hf-direct":  f"https://huggingface.co/{REPO}/resolve/main/{FILE}",
    "modelscope": f"https://modelscope.cn/models/{REPO}/resolve/master/{FILE}",
}

HDR = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}
MB = 1024 * 1024


def ranged(url, start, length, timeout=45):
    req = urllib.request.Request(url, headers={**HDR, "Range": f"bytes={start}-{start+length-1}"})
    t0 = time.time()
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        status = r.status
        while True:
            c = r.read(65536)
            if not c:
                break
            n += len(c)
    return n, time.time() - t0, status


def timed(url, nconn, total, timeout=45):
    per = max(MB, (total * MB) // nconn)
    per -= per % 65536

    def one(i):
        return ranged(url, i * per, per, timeout)

    t0 = time.time()
    try:
        if nconn == 1:
            n, _, st = one(0)
        else:
            with ThreadPoolExecutor(max_workers=nconn) as ex:
                rs = list(ex.map(one, range(nconn)))
            n = sum(r[0] for r in rs)
            st = rs[0][2]
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:60]}"
    dt = time.time() - t0
    return (n / MB / dt, st), None


if __name__ == "__main__":
    for name, url in HOSTS.items():
        print(f"\n=== {name} ===", flush=True)
        rate, err = timed(url, 1, 2)
        if err:
            print(f"  1 conn  : FAILED  {err}", flush=True)
            continue
        single, st = rate
        print(f"  1 conn  : {single:6.3f} MB/s  (HTTP {st})", flush=True)
        for k in (4, 8, 16):
            rate, err = timed(url, k, 4 * k if k <= 8 else 4 * 8)
            if err:
                print(f"  {k:2d} conns: FAILED  {err}", flush=True)
                continue
            agg, _ = rate
            print(f"  {k:2d} conns: {agg:6.3f} MB/s aggregate  "
                  f"(x{agg/max(single,1e-9):4.1f} vs 1 conn)", flush=True)
