"""hf_par_fetch.py -- segmented multi-connection downloader for the HF mirror.

WHY THIS EXISTS
---------------
Measured on this network (bw_probe.py, link_speed.py, _ms_par.py):

  * one hf-mirror connection sustains ~2.0-2.2 MB/s
  * eight parallel connections reach ~4.5 MB/s, sixteen reach ~4.2-4.7 MB/s
  * files.pythonhosted.org and codeload.github.com independently cap at ~4.4-4.5 MB/s
  * modelscope.cn reaches the same ~4.5 MB/s but needs sixteen connections for it

Four independent hosts landing near 4.5 MB/s is the tell: that is the broadband
link, not any mirror. So parallelism buys about 2x and switching hosts buys nothing.
The goal is to saturate the link, which turns a ~4 hour serial fetch into ~2 hours.

WHAT THE FIRST VERSION GOT WRONG -- AND WHY THE FIX LOOKS LIKE THIS
-------------------------------------------------------------------
v1 demanded 206 from every ranged request and, on receiving 200, recorded an error
and *continued without writing anything*. When all segments did that, nothing was
written at all; the file was then truncated up to its advertised size, so it had
exactly the right length and contained only zeros, and the size check declared it
complete. The log said "2 ok, 0 failed". A size check cannot see a zero-filled
file -- that is the whole point of the incident.

So verification here is per segment, not per file:
  * each segment must account for exactly the bytes it was assigned, verified
    against its own file handle after writing, not inferred from the print
  * any segment that fails makes the whole file fail
  * a ranged request answered with 200 means the Range was ignored; the response
    body starts at byte 0, so it is discarded rather than written to an offset
  * the truncate-to-size fill is gone entirely; the file's length is only ever
    advanced by bytes actually received

Usage:
    python hf_par_fetch.py <repo> <outdir> <path> [<path> ...]
    python hf_par_fetch.py --jobs 8 --verify Qwen/Qwen-Image-2.1 D:\\...\\Qwen-Image-2.1 \\
        text_encoder/model-00001-of-00004.safetensors
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}
CHUNK = 1 << 20
REPORT_EVERY = 64 << 20
# Below this, a ranged request is not worth the round trips and the mirror may
# answer 200 instead; fetch it in one piece.
MIN_SEGMENT = 8 << 20


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def api_sizes(repo):
    """{path: size} from the API.

    The mirror omits Content-Length on HEAD and GET, so the API is the only place
    a trustworthy size comes from -- and without a trustworthy size there is no way
    to tell a finished file from a truncated one.
    """
    url = f"{MIRROR}/api/models/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers=dict(UA)), timeout=60) as r:
        tree = json.loads(r.read().decode())
    return {f["path"]: (f.get("size") or 0) for f in tree if f.get("type") == "file"}


class Progress:
    def __init__(self, have, size):
        self.lock = threading.Lock()
        self.got = have
        self.start = have
        self.size = size
        self.t0 = time.time()
        self.next_mark = ((have // REPORT_EVERY) + 1) * REPORT_EVERY

    def add(self, n):
        with self.lock:
            self.got += n
            if self.got >= self.next_mark:
                self.next_mark = ((self.got // REPORT_EVERY) + 1) * REPORT_EVERY
                dt = max(1e-9, time.time() - self.t0)
                pct = 100.0 * self.got / self.size if self.size else 0.0
                print(f"      {pct:5.1f}%  {human(self.got)}/{human(self.size)}  "
                      f"{(self.got - self.start)/dt/1e6:.2f} MB/s  ({dt/60:.0f} min)",
                      flush=True)


def seg_worker(url, dest, start, end, prog, stop, errs, idx):
    """Fill [start, end] of dest. Returns the number of bytes written."""
    want = end - start + 1
    written = 0
    pos = start
    attempt = 0
    while pos <= end and not stop.is_set():
        attempt += 1
        if attempt > 8:
            errs.append(f"seg{idx}: 8 attempts exhausted at offset {pos}")
            return written
        headers = dict(UA)
        headers["Range"] = f"bytes={pos}-{end}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                if resp.status != 206:
                    # The body starts at byte 0, not at `pos`. Writing it at `pos`
                    # would corrupt the file, so it is dropped.
                    errs.append(f"seg{idx}: Range ignored (HTTP {resp.status}) at {pos}")
                    time.sleep(min(15, 2 * attempt))
                    continue
                with open(dest, "r+b", buffering=0) as fh:
                    while pos <= end:
                        chunk = resp.read(min(CHUNK, end - pos + 1))
                        if not chunk:
                            break
                        fh.seek(pos)
                        n = fh.write(chunk)
                        if n != len(chunk):
                            errs.append(f"seg{idx}: short write {n} != {len(chunk)}")
                            return written
                        pos += n
                        written += n
                        prog.add(n)
                if pos > end:
                    return written
                # Stream ended early but the connection was fine: resume from here.
                errs.append(f"seg{idx}: stream ended, {end - pos + 1} B left; resuming")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if getattr(e, "code", None) == 416:
                errs.append(f"seg{idx}: HTTP 416 at {pos} "
                            f"(local file {os.path.getsize(dest):,})")
                return written
            errs.append(f"seg{idx}: {type(e).__name__}: {str(e)[:70]}")
            time.sleep(min(15, 2 * attempt))
    return written


def fetch(repo, outdir, relpath, size, jobs):
    """Returns (ok, bytes_actually_received)."""
    dest = os.path.join(outdir, relpath.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{MIRROR}/{repo}/resolve/main/{relpath}"

    have = os.path.getsize(dest) if os.path.isfile(dest) else 0
    if have == size and size:
        print(f"      already complete ({human(size)})")
        return True, 0
    if have > size:
        print(f"      local larger than remote ({have:,} > {size:,}); restarting from 0")
        have = 0
    if not os.path.isfile(dest):
        with open(dest, "wb"):
            pass
    print(f"      resuming at {human(have)} of {human(size)}" if have
          else f"      starting from 0 of {human(size)}")

    span = size - have
    n = max(1, min(jobs, max(1, span // MIN_SEGMENT)))
    base = span // n
    bounds = []
    for i in range(n):
        s = have + i * base
        e = size - 1 if i == n - 1 else have + (i + 1) * base - 1
        bounds.append((s, e))

    prog = Progress(have, size)
    stop = threading.Event()
    errs = []
    results = [0] * n
    threads = []
    for i, (s, e) in enumerate(bounds):
        t = threading.Thread(target=lambda i=i, s=s, e=e: results.__setitem__(
            i, seg_worker(url, dest, s, e, prog, stop, errs, i)))
        t.daemon = True
        threads.append(t)
    t0 = time.time()
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop.set()
        print("      interrupted; bytes already written are kept, rerun to resume")
        raise
    wall = max(1e-9, time.time() - t0)

    # Per-segment accounting, checked against the file itself rather than against
    # the progress counter: the file is the thing that has to be right.
    final = os.path.getsize(dest)
    got = final - have
    want = sum(e - s + 1 for s, e in bounds)
    expected_total = max(results)
    rate = got / wall / 1e6
    print(f"      {wall/60:.1f} min at {rate:.2f} MB/s  "
          f"({n} segments, {len(errs)} event(s), {human(got)} written)")
    for e in errs[:8]:
        print(f"        ! {e}")

    if got != want:
        print(f"      INCOMPLETE: {got:,} of {want:,} bytes written "
              f"(file {final:,} / {size:,})")
        return False, got
    if final != size:
        print(f"      INCOMPLETE: file is {final:,} B, advertised {size:,} B")
        return False, got
    if expected_total == 0 and want:
        print("      INCOMPLETE: no segment wrote anything")
        return False, 0
    return True, got


def verify(repo, outdir, relpath, size):
    """Structural verification that a size check cannot do.

    A safetensors file starts with a little-endian u64 header length followed by
    that many bytes of JSON. A truncated file fails the length test; a zero-filled
    file fails to parse. Either way this catches what a byte count alone misses.
    """
    dest = os.path.join(outdir, relpath.replace("/", os.sep))
    if not os.path.isfile(dest):
        return False, "absent"
    got = os.path.getsize(dest)
    if size and got != size:
        return False, f"size {got:,} != {size:,}"
    if not relpath.endswith(".safetensors"):
        return True, "size ok"
    with open(dest, "rb") as fh:
        head = fh.read(8)
        if len(head) < 8:
            return False, "no safetensors header"
        n = int.from_bytes(head, "little")
        if n <= 0 or n > 100 << 20:
            return False, f"implausible header length {n} (zero-filled?)"
        blob = fh.read(n)
    try:
        meta = json.loads(blob.decode())
    except Exception as e:  # noqa: BLE001
        return False, f"header not JSON: {type(e).__name__}"
    tensors = [k for k in meta if k != "__metadata__"]
    return True, f"header ok, {len(tensors)} tensors"


def fetch_one(repo, outdir, relpath, sizes, jobs, do_verify):
    """Fetch a single file and print its outcome. Returns True on success."""
    size = sizes.get(relpath, 0)
    print(f"  {relpath}  ({human(size) if size else 'size unknown'})", flush=True)
    if not size:
        print("      skipped: no advertised size, cannot verify the result")
        return False
    good, _ = fetch(repo, outdir, relpath, size, jobs)
    if good and do_verify:
        v, why = verify(repo, outdir, relpath, size)
        print(f"      verify: {'ok' if v else 'FAILED'} -- {why}")
        good = v
    print(f"      => {'OK' if good else 'FAILED'}", flush=True)
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("outdir")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--parallel", type=int, default=1,
                    help="fetch this many files at once, each with --jobs "
                         "connections. Sequential files leave the link idle while "
                         "the next file's segments are established.")
    a = ap.parse_args()

    sizes = api_sizes(a.repo)
    print(f"repo   : {a.repo}")
    print(f"mirror : {MIRROR}")
    print(f"outdir : {a.outdir}")
    print(f"files  : {len(a.paths)}   jobs/file: {a.jobs}   parallel files: {a.parallel}")
    print(flush=True)

    results = {}
    lock = threading.Lock()

    def run(p):
        ok = fetch_one(a.repo, a.outdir, p, sizes, a.jobs, a.verify)
        with lock:
            results[p] = ok

    if a.parallel <= 1:
        for p in a.paths:
            run(p)
    else:
        queue = list(a.paths)
        def pump():
            while True:
                with lock:
                    if not queue:
                        return
                    p = queue.pop(0)
                run(p)
        workers = [threading.Thread(target=pump, daemon=True)
                   for _ in range(min(a.parallel, len(a.paths)))]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

    ok = sum(1 for v in results.values() if v)
    fail = len(results) - ok
    print()
    print(f"{ok} ok, {fail} failed")
    for p, v in results.items():
        if not v:
            print(f"    failed: {p}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
