"""hf_fetch.py -- fetch files from the HF mirror without huggingface_hub.

WHY NOT huggingface_hub: setting HF_ENDPOINT is not enough. The library still
issues a HEAD against huggingface.co first (recorded in the project report), and
that host times out from this network, so the download never starts. urllib with an
explicit Endpoint does not have that problem.

Also carries the two lessons this project already paid for:
  * raw.githubusercontent.com times out here; github.com, codeload, gh-proxy.com and
    hf-mirror.com do not. So GitHub content goes through a proxy, HF content through
    the mirror.
  * a bare urllib request to the HF CDN returns 403 -- a User-Agent is required.

Supports resume via HTTP Range, because these files run to gigabytes and a restart
at 90% would be its own kind of failure.

Usage:
    python hf_fetch.py <repo> <outdir> <path-in-repo> [<path> ...]
    python hf_fetch.py Qwen/Qwen-Image-2.1 D:\\work\\qwen_env\\Qwen-Image-2.1 processor/tokenizer.json
"""
import os
import sys
import time
import urllib.error
import urllib.request

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def remote_size(url):
    """Content-Length of the whole file, without downloading it."""
    req = urllib.request.Request(url, headers=dict(UA), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            n = resp.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def fetch(repo, outdir, relpath, retries=4):
    dest = os.path.join(outdir, relpath.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{MIRROR}/{repo}/resolve/main/{relpath}"

    # If the file is already complete, stop. Asking for a Range past the end is
    # what produced "HTTP 416 Requested Range Not Satisfiable", and the honest way
    # to avoid it is to compare sizes rather than to catch the error.
    want = remote_size(url)
    if os.path.isfile(dest) and want is not None and os.path.getsize(dest) == want:
        print(f"      already complete ({human(want)})")
        return dest, want

    have = os.path.getsize(dest) if os.path.isfile(dest) else 0
    if want is not None and have > want:
        print(f"      local file is larger than the remote ({have} > {want}); restarting")
        have = 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = resp.headers.get("Content-Length")
                total = (int(total) + have) if total else want
                # 206 means the server honoured the Range; 200 means it ignored it
                # and is sending the whole file, so the local part must be dropped
                # or the result would be a corrupt concatenation.
                if resp.status == 206 and have:
                    mode = "ab"
                else:
                    mode = "wb"
                    if have:
                        print("      server ignored the Range; restarting from 0")
                    have = 0
                t0 = time.time()
                got = have
                with open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                        if total and got % (32 << 20) < (1 << 20):
                            pct = 100.0 * got / total
                            rate = (got - have) / max(1e-9, time.time() - t0) / 1e6
                            print(f"      {pct:5.1f}%  {human(got)}/{human(total)}  {rate:.1f} MB/s",
                                  flush=True)
            sz = os.path.getsize(dest)
            # Verify against the advertised size. A truncated download that is not
            # checked here becomes a corrupt checkpoint that only fails much later.
            if want is not None and sz != want:
                raise OSError(f"size mismatch after download: {sz} != {want}")
            return dest, sz
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:
                # The Range start is already at or past the end. Re-check the size
                # and either accept the file or restart it.
                sz = os.path.getsize(dest) if os.path.isfile(dest) else 0
                if want is not None and sz == want:
                    print(f"      already complete ({human(sz)})")
                    return dest, sz
                print(f"      HTTP 416 and size {sz} != {want}; restarting")
                have = 0
                headers = dict(UA)
                continue
            print(f"      attempt {attempt} failed: {type(e).__name__}: {str(e)[:90]}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
            have = os.path.getsize(dest) if os.path.isfile(dest) else 0
            headers = dict(UA)
            if have:
                headers["Range"] = f"bytes={have}-"
    raise RuntimeError("unreachable")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    repo, outdir = sys.argv[1], sys.argv[2]
    paths = sys.argv[3:]
    print(f"repo    : {repo}")
    print(f"mirror  : {MIRROR}")
    print(f"outdir  : {outdir}")
    print(f"files   : {len(paths)}")
    print()
    total = 0
    for p in paths:
        print(f"  {p}")
        dest, sz = fetch(repo, outdir, p)
        total += sz
        print(f"      -> {human(sz)}")
    print()
    print(f"done, {human(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
