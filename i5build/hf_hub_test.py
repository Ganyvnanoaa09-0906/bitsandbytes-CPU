"""hf_hub_test.py -- can huggingface_hub itself do the mirror download?

The project carries a note that HF_ENDPOINT is not enough because the library
still HEADs huggingface.co, which times out on this network, so everything has
gone through hand-rolled urllib since. huggingface_hub 1.30.0 is installed and
ships multi-threaded ranged downloads (max_workers) plus hf_transfer, which is a
far better segmented downloader than anything hand-written -- if it actually
works against the mirror.

This checks exactly that, on a small file, before trusting it with 30 GB:
  1. does hf_hub_download(HF_ENDPOINT=hf-mirror) fetch at all, or hit the timeout
  2. does it agree byte-for-byte with what hf_fetch.py already produced
  3. is hf_transfer compatible with a mirror endpoint (it does its own HTTP)

Usage:
    python hf_hub_test.py
"""
import hashlib
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

REPO = "Qwen/Qwen-Image-2.1"
REL = "processor/merges.txt"
REFERENCE = r"D:\work\qwen_env\Qwen-Image-2.1\processor\merges.txt"


def md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    from huggingface_hub import hf_hub_download, constants

    print(f"huggingface_hub {__import__('huggingface_hub').__version__}")
    print(f"endpoint        : {constants.ENDPOINT}")
    print(f"hf_transfer     : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}")
    print()

    tmp = r"D:\work\_hubtest"
    os.makedirs(tmp, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=REPO,
            filename=REL,
            local_dir=tmp,
            endpoint="https://hf-mirror.com",
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: {type(e).__name__}: {str(e)[:300]}")
        return 1

    print(f"downloaded -> {path}")
    print(f"size       : {os.path.getsize(path)}")
    if os.path.isfile(REFERENCE):
        a, b = md5(path), md5(REFERENCE)
        print(f"md5 hub    : {a}")
        print(f"md5 urllib : {b}")
        print("MATCH" if a == b else "DIFFER")
    return 0


if __name__ == "__main__":
    sys.exit(main())
