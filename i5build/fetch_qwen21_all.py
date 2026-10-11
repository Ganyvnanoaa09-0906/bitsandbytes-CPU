"""fetch_qwen21_all.py -- pull every remaining file of Qwen/Qwen-Image-2.1.

WHY A DRIVER AND NOT JUST A LIST OF PATHS ON THE COMMAND LINE
-------------------------------------------------------------
The file list and the expected sizes come from the HF API rather than from a
hand-written list, because a hand-written list is what silently omits a shard and
leaves a model that loads four fifths of the way. Every file is verified against the
size the API advertises -- the mirror does not return Content-Length, so the API is
the only authoritative source available here, and a truncated multi-GB safetensors
file otherwise surfaces much later as a mysterious load error.

Order is smallest-first so that a failure 30 GB in still leaves something usable and
so the pipeline can be exercised on the small files early.

Downloads go through hf_fetch.py, which handles resume, retries, and the 416 that a
completed file produces if you ask for a Range past its end.

Usage:
    python fetch_qwen21_all.py                 # download everything missing
    python fetch_qwen21_all.py --verify-only   # just check what is on disk
"""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPO = "Qwen/Qwen-Image-2.1"
DEST = r"D:\work\qwen_env\Qwen-Image-2.1"
API = f"https://hf-mirror.com/api/models/{REPO}/tree/main?recursive=true"
UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}

# Nothing here needs the demo image or the repo boilerplate.
SKIP = ("assets/", ".gitattributes", "README.md", "LICENSE")


def api_files():
    req = urllib.request.Request(API, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        tree = json.loads(r.read().decode())
    out = []
    for f in tree:
        if f.get("type") != "file":
            continue
        p = f["path"]
        if any(p.startswith(s) or p == s for s in SKIP):
            continue
        out.append((p, f.get("size") or 0))
    return out


def human(n):
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def main():
    verify_only = "--verify-only" in sys.argv
    files = api_files()
    print(f"repo : {REPO}")
    print(f"dest : {DEST}")
    print(f"files: {len(files)}   total {human(sum(s for _, s in files))}")
    print()

    todo, have, bad = [], [], []
    for p, sz in files:
        local = os.path.join(DEST, p.replace("/", os.sep))
        if not os.path.isfile(local):
            todo.append((p, sz))
            continue
        got = os.path.getsize(local)
        if sz and got != sz:
            bad.append((p, got, sz))
            todo.append((p, sz))
        else:
            have.append((p, got))
    total_todo = sum(s for _, s in todo)
    print(f"  complete : {len(have)}")
    print(f"  missing  : {len(todo)}  ({human(total_todo)})")
    if bad:
        print(f"  WRONG SIZE (will re-fetch): {len(bad)}")
        for p, got, want in bad:
            print(f"      {p}: {got:,} != {want:,}")
    print()
    for p, sz in todo:
        print(f"    {human(sz):>9}  {p}")

    if verify_only:
        free = 0
        try:
            import shutil
            free = shutil.disk_usage(DEST).free
        except OSError:
            pass
        print()
        print(f"  disk free: {human(free)}")
        enough = free > total_todo * 1.1
        print(f"  enough for the remaining download: {enough}")
        return 0 if not todo else 1

    if not todo:
        print("nothing to do")
        return 0

    import hf_fetch

    done_bytes = 0
    failed = []
    for i, (p, sz) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {p}  ({human(sz)})", flush=True)
        try:
            path, got = hf_fetch.fetch(REPO, DEST, p)
            if sz and got != sz:
                print(f"      SIZE MISMATCH after download: {got:,} != {sz:,}")
                failed.append(p)
            else:
                done_bytes += got
                print(f"      ok  {human(got)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"      FAILED: {type(e).__name__}: {str(e)[:140]}")
            failed.append(p)

    print()
    print(f"downloaded {human(done_bytes)}; {len(failed)} failure(s)")
    for p in failed:
        print(f"    {p}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
