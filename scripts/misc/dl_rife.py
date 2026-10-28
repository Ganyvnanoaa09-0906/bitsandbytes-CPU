# -*- coding: utf-8 -*-
"""Download the RIFE interpolation assets from hf-mirror (direct huggingface.co is blocked here).

Files:
  rife_arch.py            - ComfyUI-VFI RIFE architecture (self-contained, no basicsr needed)
  rife_v4.26.safetensors  - RIFE 4.26 weights matching that architecture (~22.7 MB)
  flownet_daydream.pkl    - daydreamlive's RIFE weights (~21.4 MB), a common anime-oriented build
  README/config           - provenance
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os
import urllib.request

OUT = r"D:\work\textmodel\rife"
ITEMS = [
    ("https://hf-mirror.com/aliensmn/ComfyUI-Frame-Interpolation/resolve/main/vfi_models/rife/rife_arch.py",
     "rife_arch.py"),
    ("https://hf-mirror.com/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation/rife_v4.26.safetensors",
     "rife_v4.26.safetensors"),
    ("https://hf-mirror.com/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation/rife_v4.25.safetensors",
     "rife_v4.25.safetensors"),
    ("https://hf-mirror.com/daydreamlive/RIFE/resolve/main/flownet.pkl",
     "flownet_daydream.pkl"),
    ("https://hf-mirror.com/daydreamlive/RIFE/resolve/main/README.md",
     "README_daydream.md"),
    ("https://hf-mirror.com/daydreamlive/RIFE/resolve/main/config.json",
     "config_daydream.json"),
]


def get(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print(f"skip (exists) {os.path.basename(dst)} {os.path.getsize(dst)/1e6:.2f} MB", flush=True)
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
            total = int(r.headers.get("Content-Length", 0) or 0)
            got = 0
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                got += len(b)
        ok = (total == 0 or got == total)
        print(f"{'OK ' if ok else 'PART'} {os.path.basename(dst):<30} "
              f"{got/1e6:.2f} MB (expect {total/1e6:.2f})", flush=True)
        return ok
    except Exception as e:
        print(f"FAIL {os.path.basename(dst)}: {str(e)[:90]}", flush=True)
        if os.path.exists(dst) and os.path.getsize(dst) < 1000:
            os.remove(dst)
        return False


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for u, name in ITEMS:
        get(u, os.path.join(OUT, name))
    print("\n=== final ===", flush=True)
    for n in sorted(os.listdir(OUT)):
        print(f"  {n:<32} {os.path.getsize(os.path.join(OUT, n))/1e6:.3f} MB", flush=True)
