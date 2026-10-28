# -*- coding: utf-8 -*-
"""Wait for the 500-step LoRA run, then measure whether the LoRA changes character stability.

Chain:
  1. block until no python process is running (the training run)
  2. regenerate the SAME prompt/seed clip that produced the pre-LoRA baseline, but with the
     trained LoRA loaded
  3. run the face-consistency metric on baseline vs LoRA clip and print a comparison

Baseline: D:\\work\\output\\adiff_out\\char2_512_6f1n\\frames
  prompt = "1girl, anime style, ... gentle smile, silver hair, red eyes"
  chunk=6, steps=5, seed=1234  (independent full-schedule generation)

Usage (run with the venv python, which is the inference environment):
  python lora_ab_chain.py
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
VENV = r"D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe"
OUT = r"D:\work\output\adiff_out\lora_ab"
BASE = r"D:\work\output\adiff_out\char2_512_6f1n\frames"
LORA_DIR = r"D:\work\lora_char2_run500\final"
PROMPT = ("1girl, anime style, cute face, detailed eyes, beautiful detailed face, masterpiece, "
          "best quality, solo, upper body, gentle smile, silver hair, red eyes")


def wait_for_training():
    """Block until the training run has written its final adapter.

    NOTE: do NOT poll for "no python process" -- this script is itself a python process and
    would wait forever. Wait for the artifact instead, which is unambiguous.
    """
    target = os.path.join(LORA_DIR, "adapter_model.safetensors")
    print(f"[chain] waiting for {target} ...", flush=True)
    t0 = time.time()
    while not os.path.exists(target):
        time.sleep(30)
        if time.time() - t0 > 3 * 3600:
            print("[chain] timeout waiting for adapter; giving up", flush=True)
            return False
        if int(time.time() - t0) % 600 < 30:
            print(f"[chain]   still waiting ({(time.time()-t0)/60:.0f} min)", flush=True)
    print(f"[chain] adapter appeared after {(time.time()-t0)/60:.1f} min", flush=True)
    return True


def main():
    if not wait_for_training():
        return
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ)
    env["ADIFF_VAE"] = r"D:\work\textmodel\sd15_full\vae"

    # LoRA clip: identical prompt/seed/settings to the baseline, only the adapter differs
    lora_out = os.path.join(OUT, "with_lora")
    cmd = [VENV, "-u", os.path.join(_HERE, "independent_chunks.py"),
           "--res", "512", "--chunk", "6", "--chunks", "1", "--steps", "5",
           "--seed", "1234", "--tag", "loraAB", "--prompt", PROMPT,
           "--lora", LORA_DIR, "--lora-scale", "0.8", "--outdir", lora_out]
    print("[chain] === generating WITH lora ===", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore",
                       env=env)
    print((r.stdout or "")[-1200:], flush=True)
    if r.returncode != 0:
        print(f"[chain] lora gen failed exit={r.returncode}\n{(r.stderr or '')[-800:]}",
              flush=True)

    # face metric on both clips
    print("\n[chain] === face consistency: baseline vs lora ===", flush=True)
    for tag, d in (("baseline", BASE), ("with_lora", os.path.join(lora_out, "frames"))):
        if not os.path.isdir(d):
            print(f"  {tag}: missing {d}", flush=True)
            continue
        rr = subprocess.run([VENV, "-u", os.path.join(_HERE, "face_consistency.py"),
                             "--frames", d], capture_output=True, text=True,
                            encoding="utf-8", errors="ignore", env=env)
        print(f"--- {tag} ---", flush=True)
        print("\n".join((rr.stdout or "").strip().splitlines()[-4:]), flush=True)

    print("\n[chain] DONE", flush=True)


if __name__ == "__main__":
    main()
