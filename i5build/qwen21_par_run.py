"""qwen21_par_run.py -- drive the remaining Qwen-Image-2.1 fetch, saturated.

Measured facts this driver is built around:

  * the broadband link caps near 4.5 MB/s, shown independently by
    files.pythonhosted.org and codeload.github.com as well as by both mirrors
  * one hf-mirror connection gives ~2.0-2.2 MB/s; eight give ~4.5
  * modelscope.cn reaches the same ceiling and is kept only as a fallback

So the only lever is keeping the link busy, which has two parts: several
connections per file, and more than one file in flight so the link does not idle
while the next file's segments are being established. Three files x four
connections is the compromise -- enough concurrency to saturate, few enough
threads that a failure is still readable, and at least one file always has a
segment with data ready to read.

Partial files go first: bytes already on disk are work already paid for.

Usage:
    python qwen21_par_run.py                 # fetch everything missing
    python qwen21_par_run.py --check         # report state only
    python qwen21_par_run.py --files 3 --jobs 4
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.request

REPO = "Qwen/Qwen-Image-2.1"
DEST = r"D:\work\qwen_env\Qwen-Image-2.1"
MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
API = f"{MIRROR}/api/models/{REPO}/tree/main?recursive=true"
UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hf_par_fetch  # noqa: E402

# Not part of the model: demo assets and repository boilerplate.
SKIP = ("assets/", ".gitattributes", "README.md", "LICENSE", "configuration.json")


def human(n):
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def survey():
    with urllib.request.urlopen(urllib.request.Request(API, headers=dict(UA)), timeout=60) as r:
        tree = json.loads(r.read().decode())
    sizes = {f["path"]: (f.get("size") or 0) for f in tree if f.get("type") == "file"}
    have, todo = [], []
    for p, sz in sizes.items():
        if any(p.startswith(s) or p == s for s in SKIP):
            continue
        local = os.path.join(DEST, p.replace("/", os.sep))
        got = os.path.getsize(local) if os.path.isfile(local) else 0
        if sz and got == sz:
            have.append((p, sz))
        else:
            todo.append((p, sz, got))
    return sizes, have, todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=3, help="files in flight at once")
    ap.add_argument("--jobs", type=int, default=4, help="connections per file")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    sizes, have, todo = survey()
    remaining = sum(sz for _, sz, _ in todo) - sum(g for _, _, g in todo)
    free = shutil.disk_usage(DEST).free

    print(f"complete  : {len(have)} file(s)")
    print(f"remaining : {len(todo)} file(s), {human(remaining)} to transfer")
    print(f"disk free : {human(free)}  (needs {human(remaining)}, "
          f"headroom {human(free - remaining)})")
    print(f"concurrency: {a.files} files x {a.jobs} connections "
          f"= {a.files * a.jobs} sockets")
    print()
    # Partial files first (most bytes on disk first), then the rest largest first so
    # the longest job starts earliest and the tail is free of stragglers.
    order = sorted(todo, key=lambda t: (0, -t[2]) if t[2] else (1, -t[1]))
    for p, sz, got in order:
        mark = f"{human(got)} partial" if got else "not started"
        print(f"  {human(sz):>9}  {mark:>16}  {p}")
    print(flush=True)

    if a.check:
        return 1 if todo else 0
    if not todo:
        print("nothing to do")
        return 0

    results = {}
    lock = threading.Lock()
    queue = [p for p, _, _ in order]
    t_start = time.time()

    def pump():
        while True:
            with lock:
                if not queue:
                    return
                p = queue.pop(0)
            try:
                ok = hf_par_fetch.fetch_one(REPO, DEST, p, sizes, a.jobs, True)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"      EXCEPTION: {type(e).__name__}: {str(e)[:140]}")
                ok = False
            with lock:
                results[p] = ok
                done = len(results)
            el = (time.time() - t_start) / 60
            print(f"      [{done}/{len(order)}] done, {el:.1f} min elapsed", flush=True)

    workers = [threading.Thread(target=pump, daemon=True)
               for _ in range(min(a.files, len(queue)))]
    try:
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    except KeyboardInterrupt:
        print("\ninterrupted; partial bytes are on disk, rerun to resume")
        return 130

    ok = sum(1 for v in results.values() if v)
    fail = len(results) - ok
    print()
    print(f"{ok} ok, {fail} failed, {(time.time()-t_start)/60:.1f} min")
    for p, v in sorted(results.items()):
        if not v:
            print(f"    failed: {p}")

    # Final structural check on every large file, independent of the per-file
    # result: a correctly sized file of zeros is exactly what a size check misses.
    print()
    print("structural verification of all safetensors shards:")
    bad = 0
    for p, sz in sorted(sizes.items(), key=lambda t: -t[1]):
        if not p.endswith(".safetensors") or any(p.startswith(s) for s in SKIP):
            continue
        local = os.path.join(DEST, p.replace("/", os.sep))
        if not os.path.isfile(local):
            print(f"  ABSENT   {p}")
            bad += 1
            continue
        v, why = hf_par_fetch.verify(REPO, DEST, p, sz)
        print(f"  {'ok' if v else 'FAILED':8s} {p}  -- {why}")
        if not v:
            bad += 1
    print(f"\n{bad} bad shard(s)")
    return 1 if (fail or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
