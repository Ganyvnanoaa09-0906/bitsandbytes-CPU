"""link_speed.py -- how fast is this network path, independent of hf-mirror?

WHY THIS EXISTS
---------------
hf-mirror sustains ~2.2 MB/s, and opening eight parallel connections only lifted
the aggregate to 3.28 MB/s -- so the ceiling is not per-connection. That leaves two
very different explanations, and they call for opposite actions:

  * the broadband link itself tops out around 3-5 MB/s  -> hf-mirror is innocent,
    nothing to switch, the download simply takes the time it takes
  * the link is much faster and hf-mirror is the limit -> find a faster source

Small-file probes cannot tell these apart: a request for a 300 KB index page
measures round-trip latency, not throughput, which is exactly how the earlier
"pypi 0.19 MB/s, tuna 0.45 MB/s" figures came about. Every probe here therefore
pulls a *large* object and discards a warm-up prefix, so the reported number is
steady-state.

All endpoints are ones this project has already resolved or fetched from.

Usage:
    python link_speed.py            # all endpoints
    python link_speed.py --only pypi
"""
import argparse
import os
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}

# (name, url, bytes to pull). Sizes are deliberately >= 8 MB so that a slow start
# cannot dominate; content is discarded, only the byte count matters.
ENDPOINTS = [
    ("pypi-files", "https://files.pythonhosted.org/packages/source/n/numpy/"
                   "numpy-2.1.3.tar.gz", 12 << 20),
    ("gh-codeload", "https://codeload.github.com/pytorch/pytorch/tar.gz/refs/tags/v2.5.0",
     12 << 20),
    ("gh-proxy", "https://gh-proxy.com/https://github.com/pytorch/pytorch/archive/"
                 "refs/tags/v2.5.0.tar.gz", 12 << 20),
    ("hf-mirror", "https://hf-mirror.com/Qwen/Qwen-Image-2.1/resolve/main/"
                  "text_encoder/model-00001-of-00004.safetensors", 12 << 20),
    ("tuna-pypi", "https://pypi.tuna.tsinghua.edu.cn/packages/source/n/numpy/"
                  "numpy-2.1.3.tar.gz", 12 << 20),
]


def measure(name, url, want_bytes, warmup=2 << 20):
    """Pull `want_bytes` after discarding `warmup`, and report the steady rate."""
    try:
        req = urllib.request.Request(url, headers=dict(UA))
        with urllib.request.urlopen(req, timeout=60) as resp:
            got = 0
            while got < warmup:
                chunk = resp.read(min(1 << 20, warmup - got))
                if not chunk:
                    return (name, None, got, 0.0, "ended during warm-up")
                got += len(chunk)
            got = 0
            t0 = time.time()
            while got < want_bytes:
                chunk = resp.read(min(1 << 20, want_bytes - got))
                if not chunk:
                    break
                got += len(chunk)
            dt = time.time() - t0
        if dt <= 0 or got < (want_bytes >> 1):
            return (name, None, got, dt, f"short: {got/1e6:.1f} MB in {dt:.1f}s")
        return (name, got / dt / 1e6, got, dt, None)
    except Exception as e:  # noqa: BLE001
        return (name, None, 0, 0.0, f"{type(e).__name__}: {str(e)[:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--mb", type=int, default=12)
    a = ap.parse_args()
    want = a.mb << 20

    print(f"steady-state throughput after a 2 MB warm-up, {a.mb} MB sample each")
    print()
    rows = []
    for name, url, _ in ENDPOINTS:
        if a.only and a.only not in name:
            continue
        print(f"  probing {name:12s} ...", flush=True)
        name, rate, got, dt, err = measure(name, url, want)
        rows.append((name, rate, got, dt, err))

    print()
    print(f"  {'endpoint':14s} {'MB/s':>7s}  {'sample':>9s}  note")
    for name, rate, got, dt, err in rows:
        r = f"{rate:7.2f}" if rate else "      -"
        print(f"  {name:14s} {r}  {got/1e6:6.1f} MB  {err or ''}")
    rates = [r for _, r, _, _, _ in rows if r]
    if rates:
        print()
        print(f"  fastest: {max(rates):.2f} MB/s   slowest: {min(rates):.2f} MB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
