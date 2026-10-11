"""ms_fetch.py -- ModelScope downloader, kept as the fallback source.

Measured: modelscope.cn reaches the same ~4.5 MB/s link ceiling, but needs sixteen
connections where hf-mirror needs eight, and one connection manages 1.10 MB/s
against hf-mirror's 2.0. So it is not the faster source and it is not the primary
one. It is worth having for two concrete reasons:

  * it answers ranged requests with a clean 206 plus Content-Range, where the
    hf-mirror CDN sometimes answers 200 and ignores the Range entirely -- which is
    how a segmented downloader silently produces a correctly-sized file of zeros
  * it is an independent host serving the identical repository, so a byte-for-byte
    comparison between the two is real verification, not a self-consistency check

Both the HTTP API and the modelscope SDK path are supported; the SDK path exists
because it handles the LFS indirection itself.

Usage:
    python ms_fetch.py --list
    python ms_fetch.py --check                      # compare with what is on disk
    python ms_fetch.py --get processor/merges.txt   # one file
    python ms_fetch.py --sdk                        # whole repo via the SDK
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = "Qwen/Qwen-Image-2.1"
DEST = r"D:\work\qwen_env\Qwen-Image-2.1"
UA = {"User-Agent": "Mozilla/5.0"}
FILE_API = "https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"
TREE_API = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
CHUNK = 1 << 20


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def ms_tree(repo=REPO):
    """{path: size} for every blob in the repo, from the ModelScope API."""
    url = TREE_API.format(repo=repo)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        data = json.loads(r.read().decode())
    if data.get("Code") != 200:
        raise RuntimeError(f"ModelScope API returned {data.get('Code')}: {data.get('Message')}")
    return {f["Path"]: (f.get("Size") or 0) for f in data["Data"]["Files"]
            if f.get("Type") == "blob"}


def url_for(path, repo=REPO):
    return FILE_API.format(repo=repo, path=urllib.parse.quote(path))


def head_size(path, repo=REPO):
    """Total file size from Content-Range of a 1-byte ranged GET.

    ModelScope answers HEAD with no Content-Length, so the size has to come from a
    ranged GET -- and asking for a range also confirms the host honours ranges,
    which is the property this source is kept for.
    """
    h = dict(UA)
    h["Range"] = "bytes=0-0"
    req = urllib.request.Request(url_for(path, repo), headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        if r.status == 206 and cr and "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        return None


def get(path, dest_root=DEST, repo=REPO, size=None, jobs=8):
    dest = os.path.join(dest_root, path.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if size is None:
        size = ms_tree(repo).get(path, 0)
    have = os.path.getsize(dest) if os.path.isfile(dest) else 0
    if size and have == size:
        print(f"      already complete ({human(size)})")
        return True
    if have > size and size:
        have = 0
    if not os.path.isfile(dest):
        with open(dest, "wb"):
            pass
    print(f"      {human(have)} of {human(size)} on disk")

    span = size - have
    n = max(1, min(jobs, max(1, span // (8 << 20))))
    base = span // n
    bounds = [(have + i * base,
               size - 1 if i == n - 1 else have + (i + 1) * base - 1) for i in range(n)]

    lock = threading.Lock()
    state = {"got": have, "t0": time.time()}
    errs = []
    results = [0] * n

    def worker(i, s, e):
        pos, written = s, 0
        attempt = 0
        while pos <= e:
            attempt += 1
            if attempt > 8:
                errs.append(f"seg{i}: exhausted at {pos}")
                break
            h = dict(UA)
            h["Range"] = f"bytes={pos}-{e}"
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(url_for(path, repo), headers=h),
                        timeout=180) as r:
                    if r.status != 206:
                        errs.append(f"seg{i}: HTTP {r.status}, Range ignored")
                        time.sleep(2 * attempt)
                        continue
                    with open(dest, "r+b", buffering=0) as fh:
                        while pos <= e:
                            c = r.read(min(CHUNK, e - pos + 1))
                            if not c:
                                break
                            fh.seek(pos)
                            fh.write(c)
                            pos += len(c)
                            written += len(c)
                            with lock:
                                state["got"] += len(c)
                    if pos > e:
                        break
                    errs.append(f"seg{i}: short stream, resuming")
            except (urllib.error.URLError, TimeoutError, OSError) as ex:
                errs.append(f"seg{i}: {type(ex).__name__}: {str(ex)[:60]}")
                time.sleep(min(15, 2 * attempt))
        results[i] = written

    threads = [threading.Thread(target=worker, args=(i, s, e), daemon=True)
               for i, (s, e) in enumerate(bounds)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = max(1e-9, time.time() - t0)

    final = os.path.getsize(dest)
    want = sum(e - s + 1 for s, e in bounds)
    if final - have != want or final != size:
        print(f"      INCOMPLETE: {final:,} / {size:,} ({n} seg, {len(errs)} events)")
        for e in errs[:6]:
            print(f"        ! {e}")
        return False
    print(f"      {wall/60:.1f} min at {(final-have)/wall/1e6:.2f} MB/s "
          f"({n} segments, {len(errs)} event(s))")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--get", action="append", default=[])
    ap.add_argument("--sdk", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    if a.list or a.check:
        tree = ms_tree(a.repo)
        print(f"{len(tree)} blobs, {human(sum(tree.values()))} total")
        bad = 0
        for p, sz in sorted(tree.items(), key=lambda t: -t[1]):
            local = os.path.join(a.dest, p.replace("/", os.sep))
            got = os.path.getsize(local) if os.path.isfile(local) else 0
            if a.check:
                state = "OK" if got == sz else ("PARTIAL" if got else "missing")
                if got != sz:
                    bad += 1
                print(f"  {state:8s} {got:>13,} / {sz:>13,}  {p}")
            else:
                print(f"  {sz:>13,}  {p}")
        if a.check:
            print(f"\n{bad} file(s) not complete")
            return 1 if bad else 0
        return 0

    if a.sdk:
        from modelscope import snapshot_download
        print(f"snapshot_download({a.repo}) -> {a.dest}")
        out = snapshot_download(a.repo, local_dir=a.dest)
        print(f"done: {out}")
        return 0

    if not a.get:
        ap.print_help()
        return 2
    fails = 0
    for p in a.get:
        size = head_size(p, a.repo)
        print(f"  {p}  ({human(size) if size else 'size unknown'})")
        if not get(p, a.dest, a.repo, size, a.jobs):
            fails += 1
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
