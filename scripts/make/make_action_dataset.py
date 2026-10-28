# -*- coding: utf-8 -*-
"""Build an "action / large-motion" LoRA dataset from the frames we already rendered.

Segment 3 of the task chain is "stack an action LoRA on top of the character LoRA", so we
need a second adapter whose domain is dynamic full-body action rather than a static portrait.
Collecting external anime video is not possible here (no dataset on disk, no HF access), so
the teacher signal is our own frozen pipeline's dynamic output:

  L3_wave     stable upper-body motion      -> mild action
  L4_runjump  full-body run/jump (det 80%)  -> heavy action, the documented boundary
  L5_fight    sword swing (det 40%)         -> extreme action, collapsed
  MA_closeup  run/jump re-shot as close-up  -> action + preserved face (mitigation A)
  MB_bokeh    run/jump + DoF/bokeh          -> action + preserved anatomy (mitigation B)

The point is NOT that these frames are a good teacher (they are the model's own output, so
the adapter can only learn the model's own action style). The point is to answer the
engineering question the chain asks: can two adapters be merged into one UNet at the same
time, what combined scale is still usable, and does the stack change the collapse metrics.

Usage:
  python make_action_dataset.py --out D:\\work\\lora_action
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse
import json
import os
import shutil

MOTION = r"D:\work\output\motion"
MIT = r"D:\work\output\motion_mit"
CTL = r"D:\work\output\adiff_out"

BASE = ("1girl, anime style, cute face, detailed eyes, beautiful detailed face, "
        "masterpiece, best quality, solo")

# The ControlNet-rescued clips come first: they are the only dynamic frames on this machine
# with *correct* anatomy (the un-guided L4/L5 clips collapse), so they are the best teacher
# available for "action with intact limbs". L5_fight is deliberately excluded -- feeding the
# collapsed output back in would teach the collapse.
SOURCES = [
    (os.path.join(CTL, "ctl_L4", "on", "frames"),
     ("1girl, anime style, cute face, detailed eyes, masterpiece, best quality, "
      "solo, full body, running, jumping, dynamic action pose, energetic motion"), 10),
    (os.path.join(MOTION, "L3_wave", "frames"),
     f"{BASE}, upper body, waving hand, raised arm, greeting gesture, energetic", 10),
    (os.path.join(MIT, "MA_closeup", "frames"),
     ("1girl, anime style, cute face, detailed eyes, masterpiece, best quality, solo, "
      "close-up face, running, jumping, motion blur, speed lines, dynamic action"), 10),
    (os.path.join(MIT, "MB_bokeh", "frames"),
     ("1girl, anime style, cute face, detailed eyes, masterpiece, best quality, solo, "
      "full body, running, jumping, dynamic action pose, motion blur, bokeh, "
      "blurry background, depth of field"), 10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=10,
                    help="how many frames to take from each source dir")
    a = ap.parse_args()

    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    meta, n = {}, 0
    for src, prompt, _ in SOURCES:
        if not os.path.isdir(src):
            print(f"[skip] missing {src}")
            continue
        files = sorted(f for f in os.listdir(src) if f.lower().endswith(".png"))
        tag = os.path.basename(os.path.dirname(src))
        for i, f in enumerate(files[:a.per_source]):
            name = f"{tag}_{i:02d}.png"
            shutil.copyfile(os.path.join(src, f), os.path.join(img_dir, name))
            meta[name] = {"prompt": prompt, "source": tag}
            n += 1
        print(f"[ok] {tag}: {min(len(files), a.per_source)} frames")

    with open(os.path.join(a.out, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] {n} images + prompts.json -> {a.out}")


if __name__ == "__main__":
    main()
