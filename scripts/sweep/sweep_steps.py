# -*- coding: utf-8 -*-
"""Sweep the number of denoising steps for chunk 0 (the full-schedule chunk).

Chunk 0 dominates cost (6 steps) while carried chunks run only 3 (t=500/333/166).
This measures the quality cost of dropping chunk-0 steps, i.e. whether a "fast"
profile is viable: lower steps -> proportionally less time, at some MAD cost.

Everything else is held fixed: 512, chunk=10, overlap=2, cn=500, DPM-Solver++(2M),
Counterfeit-V3 + v1-5, same prompt/seed.

For each steps value we record: overall MAD, intra MAD, boundary MAD, hist drift,
elapsed time, and seconds/frame.

Usage:
  python sweep_steps.py --res 512 --chunk 10 --overlap 2 --chunks 2 --steps-list 4,5,6
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.anime_adiff import build, ANIME_PROMPT, ANIME_NEG, OUT_DIR
from scripts.lib.sweep_carry_noise import generate, measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--steps-list", default="4,5,6")
    ap.add_argument("--carry-noise", type=float, default=500.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="stepscan")
    ap.add_argument("--prompt", default=ANIME_PROMPT)
    ap.add_argument("--neg", default=ANIME_NEG)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    steps_list = [int(x) for x in a.steps_list.split(",") if x.strip()]

    print(f"[cfg] res={a.res} chunk={a.chunk} overlap={a.overlap} chunks={a.chunks} "
          f"carry-noise={a.carry_noise} steps sweep={steps_list} "
          f"solver=DPM-Solver++(2M) base=counterfeit-v3", flush=True)

    pipe = build("dpm")
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, "cpu", 1, True, negative_prompt=a.neg)
    embeds = torch.cat([ne, pe])

    gprobe = torch.Generator().manual_seed(a.seed)
    torch.randn((1, 4, a.chunk, a.res // pipe.vae_scale_factor,
                 a.res // pipe.vae_scale_factor), generator=gprobe)
    gen_state = {"seed": a.seed, "warm": gprobe.get_state().clone()}

    results = []
    for st in steps_list:
        print(f"\n{'='*60}\n[run] chunk0 steps = {st}\n{'='*60}", flush=True)
        # NOTE: carry-noise maps onto the schedule, so a shorter schedule changes which
        # timesteps the carried chunks run. Log it so the comparison stays honest.
        frames, dt, log = generate(pipe, embeds, dict(gen_state), a.res, a.chunk,
                                   a.overlap, a.chunks, st, a.carry_noise,
                                   a.prompt, a.neg)
        m = measure(frames, a.chunk, a.overlap)
        m.update(steps=st, seconds=dt, frames=len(frames),
                 sec_per_frame=dt / len(frames))
        results.append(m)
        print(f"[result] steps={st} mad={m['mad']:.2f} intra={m['intra']:.2f} "
              f"boundary={m['boundary']:.2f} hist={m['hist']:.4f} "
              f"time={dt:.1f}s ({m['sec_per_frame']:.1f}s/frame)", flush=True)
        print(f"         {'; '.join(log)}", flush=True)

    print(f"\n\n{'#'*64}")
    print(f"# STEP SWEEP SUMMARY (res={a.res}, chunk={a.chunk}, chunks={a.chunks}, "
          f"cn={a.carry_noise:.0f})")
    print(f"{'#'*64}")
    print(f"{'chunk0 steps':>12} {'overall MAD':>12} {'intra':>8} {'boundary':>9} "
          f"{'hist':>8} {'time(s)':>9} {'s/frame':>8}")
    for r in results:
        print(f"{r['steps']:>12} {r['mad']:>12.2f} {r['intra']:>8.2f} "
              f"{r['boundary']:>9.2f} {r['hist']:>8.4f} {r['seconds']:>9.1f} "
              f"{r['sec_per_frame']:>8.1f}")

    # quality cost per step saved, relative to the largest step count
    base = max(results, key=lambda r: r["steps"])
    print(f"\n--- relative to steps={base['steps']} ---")
    for r in sorted(results, key=lambda x: x["steps"]):
        if r["steps"] == base["steps"]:
            continue
        saved = 1 - r["seconds"] / base["seconds"]
        dmad = (r["mad"] - base["mad"]) / base["mad"] * 100
        print(f"  steps={r['steps']}: time {saved*100:+.1f}%  "
              f"overall MAD {dmad:+.1f}%  hist {(r['hist']-base['hist'])/base['hist']*100:+.1f}%")

    out = os.path.join(OUT_DIR, f"{a.tag}_{a.res}_stepssweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(config=dict(res=a.res, chunk=a.chunk, overlap=a.overlap,
                                   chunks=a.chunks, carry_noise=a.carry_noise,
                                   steps_list=steps_list), results=results), f, indent=2)
    print(f"\n[save] {out}", flush=True)


if __name__ == "__main__":
    main()
