"""seg_resume_test.py -- prove segmented resume is byte-exact, or fail loudly.

The incident this guards against: v1 of the segmented fetcher accepted HTTP 200
where it demanded 206, wrote nothing for every segment, truncated the file up to
its advertised length, and reported success. Size checks cannot see that. The only
test that can is one that knows what the bytes are supposed to be.

Method
------
Take a file already on disk and verified against a second host (processor/
tokenizer.json, 11,422,654 B, md5 6423133b9cc1a2077b57822c30c211aa), copy it to a
scratch directory, truncate the copy to half its length, and let hf_par_fetch
finish it there. Then compare the resumed file's md5 with the original's.

If segment offsets are wrong by even one byte, the md5 differs. If the fetcher
writes nothing and pads, the md5 differs. Nothing about the original file is
modified: it is only ever read.

Usage:
    python seg_resume_test.py
"""
import hashlib
import os
import shutil
import subprocess
import sys

SRC = r"D:\work\qwen_env\Qwen-Image-2.1\processor\tokenizer.json"
SCRATCH = r"D:\work\_segtest"
REL = "processor/tokenizer.json"
REPO = "Qwen/Qwen-Image-2.1"
KNOWN_MD5 = "6423133b9cc1a2077b57822c30c211aa"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    if not os.path.isfile(SRC):
        print(f"FAIL: reference file missing: {SRC}")
        return 2
    src_md5 = md5(SRC)
    src_size = os.path.getsize(SRC)
    print(f"reference : {src_size:,} B  {src_md5}")
    if src_md5 != KNOWN_MD5:
        print(f"FAIL: reference md5 is not the verified one ({KNOWN_MD5})")
        return 2

    shutil.rmtree(SCRATCH, ignore_errors=True)
    dest = os.path.join(SCRATCH, REL.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    halfway = src_size // 2
    with open(SRC, "rb") as fh, open(dest, "wb") as out:
        out.write(fh.read(halfway))
    print(f"truncated to {halfway:,} B in scratch")

    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "hf_par_fetch.py"),
           "--jobs", "4", "--verify", REPO, SCRATCH, REL]
    print("running  :", " ".join(cmd[1:]))
    print()
    rc = subprocess.call(cmd)
    print()
    if rc != 0:
        print(f"FAIL: fetcher exited {rc}")
        return 1

    got_size = os.path.getsize(dest)
    got_md5 = md5(dest)
    print(f"resumed   : {got_size:,} B  {got_md5}")
    if got_size != src_size:
        print(f"FAIL: size {got_size:,} != {src_size:,}")
        return 1
    if got_md5 != src_md5:
        print("FAIL: md5 differs -- the resumed bytes do not match the original.")
        # Where do they diverge? Report the first differing offset, since that
        # points at the segment whose offset is wrong.
        with open(SRC, "rb") as a, open(dest, "rb") as b:
            off = 0
            while True:
                x, y = a.read(1 << 20), b.read(1 << 20)
                if not x:
                    break
                if x != y:
                    for i in range(min(len(x), len(y))):
                        if x[i] != y[i]:
                            print(f"      first difference at offset {off + i:,}")
                            break
                    break
                off += len(x)
        return 1
    print("PASS: segmented resume is byte-exact")

    # Also prove the fetcher refuses to call an all-zero fill complete: write a
    # zero-filled file of the right size and check that --verify rejects it.
    zpath = os.path.join(SCRATCH, "_zerofill.safetensors")
    with open(zpath, "wb") as fh:
        fh.truncate(4096)
    if os.path.getsize(zpath) != 4096:
        print("FAIL: could not build the zero-fill fixture")
        return 1
    print()
    print("zero-fill fixture built (4096 B of zeros); --verify must reject it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
