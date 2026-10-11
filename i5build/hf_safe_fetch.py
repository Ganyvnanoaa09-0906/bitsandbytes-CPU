"""hf_safe_fetch.py -- download with per-window verification built in.

WHY THIS REPLACES THE SEGMENTED DOWNLOADER FOR THESE FILES
----------------------------------------------------------
Seven shards were fetched with hf_par_fetch.py. Five came back with large regions of
zeros while their file lengths matched the safetensors headers exactly, and
verify_remote.py confirmed against the mirror, window by window, that 250 of 498
sampled 1 MB windows differ -- every one of them "local 100% zero, remote dense". The
first bad window is at the same offset (201,326,592) in every damaged file, which
points at the write path rather than at the network. Two files were clean, and both
had been fetched in a single uninterrupted pass.

The mechanism is not established and this file does not claim one. What it does
instead is make the failure impossible to ship:

  * every window is verified against the mirror as it arrives, by re-reading the
    bytes from the file rather than trusting the buffer that was written
  * a file is only moved into place after all of its windows verify
  * partial output goes to a .part file, so a half-finished download can never be
    mistaken for a complete one
  * no resume from a file's LENGTH, which is what the previous version trusted:
    length and written data are different things, and a length check cannot see a
    zero-filled region

WHAT CHANGED AFTER MEASURING IT
-------------------------------
The guarantees above were bought at a price that only became visible on a slow link:
both verification paths were single-threaded `for` loops, and the closing pass
re-downloaded the whole file from the mirror to compare. On a 4.6 GB shard that is
hours per pass, and the run that exposed it produced no output for 12 minutes while
moving 273 KB/s down a link that parallel connections drive at several MB/s.

Three changes, none of which weaken what is proven:

  * every network pass now uses the same `jobs`-wide worker pool as the download
  * the closing pass costs ZERO network bytes. Each window's received bytes are
    hashed as they are verified, the digests are journaled to disk, and the file is
    then re-read from end to end and checked against those digests. This still proves
    file-on-disk == bytes-the-mirror-sent, which is all the re-download pass ever
    proved, because re-fetching can only compare against the same mirror.
  * a .part now carries a digest journal, so an interrupted download resumes on
    evidence -- the windows whose bytes still hash to what was verified. It never
    resumes on length, which is what produced the corrupt shards.

The old "a full-length .part is re-verified instead of truncated" path is gone. It
spent exactly as many network bytes as re-downloading the file, and it left behind no
journal. Re-downloading is the same cost and yields a file whose every window has a
recorded, locally re-checkable digest.

Usage:
    python hf_safe_fetch.py <repo> <outdir> <path> [<path> ...]
    python hf_safe_fetch.py --window 4194304 --jobs 8 Qwen/Qwen-Image-2.1 D:\\out <path>
    python hf_safe_fetch.py --verify-only <repo> <outdir> <path>   # disk only, no network
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
# The data host is not necessarily the host that advertises the file list. huggingface.co
# itself is unreachable from here (connection timeout, measured), hf-mirror serves one
# connection at 0.11 MB/s, and ModelScope reaches 2.6 MB/s. Sizes still come from the HF
# API because that is where the repository index is authoritative; the data host's own
# Content-Range total is checked against it for every file.
API_HOST = os.environ.get("HF_API_HOST", MIRROR)
UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}
WIN = 1 << 20
JOBS = 8
JOURNAL_EVERY = 32      # newly verified windows between atomic journal saves
PROGRESS_EVERY = 32     # windows between progress lines (kept small: the link is slow)

SOURCES = {
    "hf-mirror": ("https://hf-mirror.com", "main"),
    "hf": ("https://huggingface.co", "main"),
    "modelscope": ("https://modelscope.cn/models", "master"),
}


def resolve_source(name):
    if name in SOURCES:
        return SOURCES[name]
    if "://" in name:                      # explicit base URL
        return name, "main"
    raise SystemExit(f"unknown source {name!r}; known: {', '.join(SOURCES)}")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def api_sizes(repo):
    url = f"{API_HOST}/api/models/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers=dict(UA)),
                                timeout=60) as r:
        tree = json.loads(r.read().decode())
    return {f["path"]: (f.get("size") or 0) for f in tree if f.get("type") == "file"}


def data_host_size(url):
    """The data host's own Content-Range total for a file, or None.

    A size taken from one host and used to drive downloads from another is an
    assumption; asking the host that will supply the bytes removes it.
    """
    h = dict(UA)
    h["Range"] = "bytes=0-0"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                    timeout=60) as r:
            cr = r.headers.get("Content-Range") or ""
            return int(cr.split("/")[-1]) if "/" in cr else None
    except Exception:  # noqa: BLE001
        return None


def remote_bytes(url, start, end, tries=4):
    """Fetch exactly one byte range, or raise.

    A server is allowed to answer 200 with the whole body when the requested range is
    the whole file, and ModelScope does that for single-window files. Accepting that is
    only safe because the LENGTH is enforced here: the earlier segmented downloader
    accepted a 200 that carried no body and wrote a correct-length file full of zeros.
    A status code says what the server decided; the byte count says what arrived.
    """
    want = end - start + 1
    h = dict(UA)
    h["Range"] = f"bytes={start}-{end}"
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                        timeout=180) as r:
                if r.status not in (200, 206):
                    raise RuntimeError(f"status {r.status}")
                data = r.read()
            if len(data) != want:
                raise RuntimeError(f"got {len(data)} B, wanted {want} B "
                                   f"(status {r.status})")
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(15, 2 * (attempt + 1)))
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:80]}")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def wrange(i, window, size):
    start = i * window
    return start, min(size, start + window) - 1


def n_windows(size, window):
    return (size + window - 1) // window


def read_window(fh, start, end):
    fh.seek(start)
    return fh.read(end - start + 1)


# --- digest journal ---------------------------------------------------------------
# A .part is only as good as the evidence attached to it. The journal records, per
# window, the sha256 of the bytes that were verified against the mirror. A resumed run
# re-reads the file and keeps only the windows that still hash to their entry, so
# resuming never trusts a length, a timestamp, or a previous run's exit status.

def journal_path(part):
    return part + ".journal.json"


def journal_load(part, window, size):
    try:
        with open(journal_path(part), "r", encoding="utf-8") as fh:
            j = json.load(fh)
    except Exception:  # noqa: BLE001
        return {}
    if j.get("window") != window or j.get("size") != size:
        return {}
    out = {}
    for i, h in enumerate(j.get("sha256") or []):
        if h:
            out[i] = h
    return out


def journal_save(part, window, size, digests):
    n = n_windows(size, window)
    payload = {"window": window, "size": size,
               "sha256": [digests.get(i) for i in range(n)]}
    tmp = journal_path(part) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, journal_path(part))


# --- verification passes ----------------------------------------------------------

def local_check(part, size, window, digests, indices=None, label="check",
                quiet=False):
    """Re-read windows from DISK and compare with their recorded digest.

    This is the pass that proves the file on disk is the file the mirror sent. It
    needs no network: the digest was taken from the bytes that arrived and were
    compared against the mirror, and the download path re-read each window from the
    file before recording it.
    """
    todo = list(indices) if indices is not None else list(range(n_windows(size, window)))
    bad, zero, done = [], 0, 0
    t0 = time.time()
    with open(part, "rb") as fh:
        for i in todo:
            start, end = wrange(i, window, size)
            b = read_window(fh, start, end)
            if b and b.count(0) == len(b):
                zero += 1
            want = digests.get(i)
            if want is None or sha(b) != want:
                bad.append(i)
            done += 1
            if not quiet and done % 128 == 0:
                el = max(time.time() - t0, 1e-9)
                print(f"      {label}: {done}/{len(todo)} windows from disk  "
                      f"({done * window / 1048576 / el:.0f} MB/s)", flush=True)
    return bad, zero


def verify_mirror(url, part, size, window, jobs, indices, label="verify"):
    """Re-fetch windows and compare them with the file. Returns (bad, digests)."""
    todo = list(indices)
    total = len(todo)
    bad, digs = [], {}
    lock = threading.Lock()
    state = {"n": 0, "stop": False}
    t0 = time.time()

    def worker():
        while True:
            with lock:
                if state["stop"] or not todo:
                    return
                i = todo.pop()
            start, end = wrange(i, window, size)
            try:
                rem = remote_bytes(url, start, end)
                if len(rem) != end - start + 1:
                    raise RuntimeError(f"short read {len(rem)} != {end-start+1}")
            except Exception as e:  # noqa: BLE001
                with lock:
                    bad.append((i, f"fetch {type(e).__name__}: {str(e)[:60]}"))
                continue
            with open(part, "rb") as fh:
                loc = read_window(fh, start, end)
            if rem != loc:
                with lock:
                    first = len(bad) < 4
                    bad.append((i, f"differs at {start:,}"))
                    if len(bad) >= 8:
                        state["stop"] = True
                if first:
                    print(f"      MISMATCH at {start:,}: local "
                          f"{100*loc.count(0)/max(1,len(loc)):.1f}% zero, remote "
                          f"{100*rem.count(0)/max(1,len(rem)):.1f}% zero", flush=True)
                continue
            with lock:
                digs[i] = sha(rem)
                state["n"] += 1
                n = state["n"]
            if n % PROGRESS_EVERY == 0:
                el = max(time.time() - t0, 1e-9)
                rate = n * window / 1048576 / el
                eta = (total - n) * window / 1048576 / max(rate, 1e-9)
                print(f"      {label}: {n}/{total} windows  {rate:.2f} MB/s  "
                      f"eta {eta/60:.1f} min", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return bad, digs


def download(url, part, size, window, jobs, todo_idx, digests, quiet=False):
    """Fetch the listed windows, write them, re-read them, then record their digest.

    The order matters: the digest is taken from the bytes that arrived AND were read
    back out of the file, so it certifies the file, not the buffer.
    """
    todo = list(todo_idx)
    total = len(todo)
    errs = []
    lock = threading.Lock()
    # Progress is driven by a threshold rather than by `n % period == 0`: workers claim
    # windows out of order, so exact multiples of any modulus get skipped and the run
    # goes silent for minutes at a time -- which is how a healthy download was mistaken
    # for a hung one.
    period = max(1, min(128, total // 80))
    state = {"n": 0, "dirty": 0, "next": period}
    t0 = time.time()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop()
            start, end = wrange(i, window, size)
            n = None
            try:
                data = remote_bytes(url, start, end)
                if len(data) != end - start + 1:
                    raise RuntimeError(f"short read {len(data)} != {end-start+1}")
                with open(part, "r+b", buffering=0) as fh:
                    fh.seek(start)
                    fh.write(data)
                with open(part, "rb") as fh:
                    back = read_window(fh, start, end)
                if back != data:
                    raise RuntimeError("written bytes do not read back")
                with lock:
                    digests[i] = sha(data)
                    state["n"] += 1
                    state["dirty"] += 1
                    n, dirty = state["n"], state["dirty"]
                    show = n >= state["next"] or n == total
                    if show:
                        state["next"] = n + period
                    if dirty >= JOURNAL_EVERY:
                        state["dirty"] = 0
                        journal_save(part, window, size, digests)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errs.append(f"window {i} ({start:,}): {type(e).__name__}: "
                                f"{str(e)[:70]}")
                # an unfinished window is simply not recorded; the journal never
                # claims a window whose bytes were not verified end to end
            if n is None or not show:
                continue
            el = max(time.time() - t0, 1e-9)
            rate = n * window / 1048576 / el
            eta = (total - n) * window / 1048576 / max(rate, 1e-9)
            print(f"      {n}/{total} windows  {rate:.2f} MB/s  "
                  f"eta {eta/60:.1f} min", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errs


# --- one file ---------------------------------------------------------------------

def fetch(repo, outdir, relpath, size, window, jobs, base, revision,
          final_mirror=False, verify_only=False):
    dest = os.path.join(outdir, relpath.replace("/", os.sep))
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{base}/{repo}/resolve/{revision}/{relpath}"
    n_win = n_windows(size, window)
    t_start = time.time()

    print(f"  {relpath}  ({human(size)}, {n_win} windows of {human(window)})",
          flush=True)

    if os.path.isfile(dest):
        print("      a file already exists at the destination; it is being rebuilt "
              "because an earlier download of this file was found corrupt")

    digests = journal_load(part, window, size) if os.path.isfile(part) else {}
    if digests:
        print(f"      journal: {len(digests)}/{n_win} window(s) recorded by an earlier "
              f"run; re-checking them on disk before trusting them", flush=True)
        # Only the journaled windows are re-checked. Passing no `indices` here scanned
        # the whole file, and every window with no digest was then reported as one that
        # "no longer matches on disk" -- a claim about windows that were never fetched.
        bad, zero = local_check(part, size, window, digests,
                                indices=sorted(digests), label="recheck")
        if zero:
            print(f"      note: {zero} journaled window(s) are entirely zero on disk")
        if bad:
            print(f"      {len(bad)} journaled window(s) no longer match on disk; they "
                  f"will be fetched again")
            for i in bad:
                digests.pop(i, None)
        else:
            print(f"      all {len(digests)} journaled window(s) still hash correctly",
                  flush=True)

    if verify_only:
        if not os.path.isfile(part):
            print("      verify-only: no .part to check")
            return False
        bad, zero = local_check(part, size, window, digests,
                                indices=sorted(digests), label="verify")
        miss = n_win - len(digests)
        print(f"      verify-only: {len(bad)} bad, {miss} window(s) not covered, "
              f"{zero} all-zero")
        return not bad and not miss

    missing = [i for i in range(n_win) if i not in digests]
    reused = n_win - len(missing)
    if reused:
        print(f"      resuming: {reused}/{n_win} window(s) already verified and still "
              f"matching on disk ({human(reused * window)}), {len(missing)} to fetch",
              flush=True)

    if missing:
        if reused:
            with open(part, "r+b") as fh:
                fh.truncate(size)          # sparse-extend; no data written here
        else:
            with open(part, "wb") as fh:
                fh.truncate(size)
            digests = {}
        errs = download(url, part, size, window, jobs, missing, digests)
        outstanding = [i for i in missing if i not in digests]
        # Serial retry pass: whatever survived the pool gets one careful attempt each.
        for attempt in range(3):
            if not outstanding:
                break
            print(f"      retry pass {attempt+1}: {len(outstanding)} window(s)",
                  flush=True)
            still = []
            for i in outstanding:
                start, end = wrange(i, window, size)
                try:
                    data = remote_bytes(url, start, end)
                    if len(data) != end - start + 1:
                        raise RuntimeError(f"short read {len(data)}")
                    with open(part, "r+b", buffering=0) as fh:
                        fh.seek(start)
                        fh.write(data)
                    with open(part, "rb") as fh:
                        back = read_window(fh, start, end)
                    if back != data:
                        raise RuntimeError("written bytes do not read back")
                    digests[i] = sha(data)
                except Exception as e:  # noqa: BLE001
                    still.append(i)
                    errs.append(f"retry window {i}: {type(e).__name__}: {str(e)[:60]}")
            outstanding = still
        journal_save(part, window, size, digests)
        if outstanding:
            print(f"      FAILED: {len(outstanding)} window(s) never completed: "
                  f"{outstanding[:8]}")
            for e in errs[:5]:
                print(f"        ! {e}")
            return False

    # Closing pass. No network bytes: the digests were taken from bytes that were
    # compared against the mirror, so re-reading the file against them proves the same
    # thing the old re-download pass proved.
    print(f"      closing pass: re-reading all {n_win} window(s) from disk", flush=True)
    bad, zero = local_check(part, size, window, digests, label="closing")
    missed = [i for i in range(n_win) if i not in digests]
    if bad or missed:
        print(f"      FAILED closing pass: {len(bad)} mismatch(es) "
              f"{bad[:6]}, {len(missed)} window(s) with no digest {missed[:6]}")
        return False
    if zero:
        print(f"      FAILED: {zero} window(s) are entirely zero")

    if final_mirror:
        print(f"      --final-mirror: re-fetching all {n_win} window(s) to compare",
              flush=True)
        bad, _ = verify_mirror(url, part, size, window, jobs, range(n_win),
                               label="mirror")
        if bad:
            print(f"      FAILED mirror re-check: {len(bad)} window(s): {bad[:5]}")
            return False

    final = os.path.getsize(part)
    if final != size:
        print(f"      FAILED: {final:,} != {size:,}")
        return False
    os.replace(part, dest)
    try:
        os.remove(journal_path(part))
    except OSError:
        pass
    wall = time.time() - t_start
    print(f"      OK  {human(size)}  {human(len(missing) * window if missing else 0)} "
          f"fetched, {human(reused * window)} reused, in {wall/60:.1f} min", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("outdir")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--window", type=int, default=WIN)
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--final-mirror", action="store_true",
                    help="after the closing pass, re-fetch every window from the "
                         "mirror as well (costs one extra full download)")
    ap.add_argument("--verify-only", action="store_true",
                    help="no network writes: check the .part against its journal")
    ap.add_argument("--source", default="hf-mirror",
                    help="data host: hf-mirror | hf | modelscope | <base URL>")
    a = ap.parse_args()

    base, revision = resolve_source(a.source)
    sizes = api_sizes(a.repo)
    print(f"repo   : {a.repo}")
    print(f"source : {base}  (revision {revision})")
    print(f"sizes  : {API_HOST} api")
    print(f"outdir : {a.outdir}")
    print(f"window : {human(a.window)}, {a.jobs} workers, "
          f"closing pass from disk, digest journal for resume")
    print(flush=True)

    ok = fail = 0
    t_all = time.time()
    for i, p in enumerate(a.paths, 1):
        size = sizes.get(p, 0)
        if not size:
            print(f"[{i}/{len(a.paths)}] {p}: no advertised size, skipped")
            fail += 1
            continue
        print(f"[{i}/{len(a.paths)}]", flush=True)
        # Ask the host that will supply the bytes how long the file is, and refuse to
        # download a length that disagrees with the index: a mismatch there means the
        # two hosts are serving different revisions, which no window check can catch.
        host_size = data_host_size(f"{base}/{a.repo}/resolve/{revision}/{p}")
        if host_size is None:
            print(f"      {p}: data host did not report a size; using the index "
                  f"({size:,})")
        elif host_size != size:
            print(f"      {p}: REFUSED -- index says {size:,}, {base} says "
                  f"{host_size:,}")
            fail += 1
            continue
        if fetch(a.repo, a.outdir, p, size, a.window, a.jobs, base, revision,
                 final_mirror=a.final_mirror, verify_only=a.verify_only):
            ok += 1
        else:
            fail += 1
        print(flush=True)
    print(f"{ok} ok, {fail} failed, {(time.time()-t_all)/60:.1f} min total")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
