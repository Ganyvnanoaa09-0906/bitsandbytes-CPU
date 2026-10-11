"""fetch_qwen21_par.py -- finish the Qwen-Image-2.1 fetch with parallel segments.

Companion to fetch_qwen21_all.py. That one walks the API tree and pulls the
remaining files serially through hf_fetch.py, which measured 2.2 MB/s. This one
pulls them through hf_par_fetch.py, whose eight-way segmentation reaches about
4.2 MB/s -- the broadband link's own ceiling, established by finding that
files.pythonhosted.org and codeload.github.com both cap at ~4.5 MB/s independently.

Ordering rule: whatever already has bytes on disk goes first. A partially
downloaded 5 GB shard is the one piece of work that is already paid for, and
finishing it converts sunk bytes into a usable file sooner.
"""
import json
import os
import shutil
import sys
import time
import urllib.request

REPO = "Qwen/Qwen-Image-2.1"
DEST = r"D:\work\qwen_env\Qwen-Image-2.1"
MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
API = f"{MIRROR}/api/models/{REPO}/tree/main?recursive=true"
UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}
JOBS = 10

# hf_par_fetch.py lives beside this file.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hf_par_fetch  # noqa: E402


def human(n):
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def main():
    with urllib.request.urlopen(urllib.request.Request(API, headers=dict(UA)), timeout=60) as r:
        tree = json.loads(r.read().decode())
    files = [(f["path"], f.get("size") or 0) for f in tree
             if f.get("type") == "file"
             and not f["path"].startswith("assets/")
             and f["path"] not in (".gitattributes", "README.md", "LICENSE")]

    todo, have = [], []
    for p, sz in files:
        local = os.path.join(DEST, p.replace("/", os.sep))
        got = os.path.getsize(local) if os.path.isfile(local) else 0
        if sz and got == sz:
            have.append((p, sz))
        else:
            todo.append((p, sz, got))

    total = sum(sz for _, sz, _ in todo) - sum(g for _, _, g in todo)
    free = shutil.disk_usage(DEST).free
    print(f"complete : {len(have)}")
    print(f"remaining: {len(todo)} files, {human(total)} to transfer")
    print(f"disk free: {human(free)}   (needed {human(total)}, headroom {human(free - total)})")
    print()
    for p, sz, got in sorted(todo, key=lambda t: (-t[2], t[1])):
        mark = f"{human(got)} partial" if got else "not started"
        print(f"  {human(sz):>9}  {mark:>16}  {p}")
    print()

    # Partial files first, then the rest smallest-first so a failure late still
    # leaves a model that can be inspected.
    order = sorted(todo, key=lambda t: (0 if t[2] else 1, t[1] if not t[2] else 0))
    ok = fail = 0
    t0 = time.time()
    for i, (p, sz, got) in enumerate(order, 1):
        print(f"[{i}/{len(order)}] {p}  ({human(sz)})", flush=True)
        try:
            if hf_par_fetch.fetch(REPO, DEST, p, sz, JOBS):
                ok += 1
            else:
                fail += 1
        except KeyboardInterrupt:
            print("interrupted; rerun to resume")
            break
        except Exception as e:  # noqa: BLE001
            print(f"      FAILED: {type(e).__name__}: {str(e)[:140]}")
            fail += 1
        el = time.time() - t0
        print(f"      elapsed {el/60:.1f} min", flush=True)

    print()
    print(f"{ok} ok, {fail} failed, {(time.time()-t0)/60:.1f} min")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
