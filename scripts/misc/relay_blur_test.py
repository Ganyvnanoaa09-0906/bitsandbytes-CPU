# -*- coding: utf-8 -*-
"""Verify the accumulated-smoothing hypothesis, and test the fixed initialisation.

Hypothesis: the relay initialises the 8 NEW frames as copies of the previous chunk's last
frame (prev_tail[:, :, -1:].repeat(...)) and then runs only 2 denoising steps. Because the
solver believes it is at t=400 while the actual noise is far lower, each relay behaves like
a low-pass filter. Result: detail decays chunk over chunk (measured: 512-run Laplacian
variance 423.8 -> 26.8, a 94% loss), frames inside a relay chunk stay near-identical
(hence the deceptively low MAD 1.0-1.5), and the clip looks like an image-to-image
slideshow.

Two initialisations compared, everything else identical (256, chunk=10, overlap=2,
chunks=3, steps=5, relay runs the last 2 schedule steps):
  A "repeat-pad" : current behaviour -- new frames = repeat of last carried frame.
  B "independent": new frames = fresh standard normal noise; carried frames are noised
                   with the scheduler's native add_noise to the relay timestep so the whole
                   batch sits at a comparable noise level (this is what lets the motion
                   module compute real inter-frame motion instead of treating frames as
                   static duplicates).

Reports per chunk: Laplacian variance (detail), MAD, white%, green_px.

Usage:
  python relay_blur_test.py --res 256 --chunk 10 --overlap 2 --chunks 3 --steps 5 --relay-steps 2
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
import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.anime_adiff import build, ANIME_PROMPT, ANIME_NEG

GUIDANCE = 7.5


def lapvar(a):
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def frame_stats(a):
    # cast FIRST: uint8 subtraction underflows (g - max(r,b) wraps to ~255 when g < r),
    # which silently produced nonsense green/blown counts in the first run of this test.
    a = a.astype(np.float32)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx = a.max(2)
    mn = a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    green = g - np.maximum(r, b)
    return dict(lap=lapvar(a), white=float(100 * (a > 245).mean()),
                blown=float(((lum > 235) & (sat < 0.08)).mean() * 100),
                green_px=int((green > 25).sum()),
                mean=float(a.mean()))


def run(pipe, embeds, seed, res, chunk, overlap, chunks, steps, relay_steps,
        t_start, mode, prompt, neg, noise_t="match"):
    lat_h = res // pipe.vae_scale_factor
    sched = pipe.scheduler
    gen = torch.Generator().manual_seed(seed)
    torch.randn((1, 4, chunk, lat_h, lat_h), generator=gen)

    prev_tail = None
    per_chunk = []
    all_frames = []

    for ci in range(chunks):
        is_first = ci == 0
        sched.set_timesteps(steps)
        ts = sched.timesteps
        if is_first:
            skip = 0
        else:
            skip = max(0, len(ts) - relay_steps)
        sched._step_index = skip
        sched.model_outputs = [None] * sched.config.solver_order
        sched._lower_order_nums = 0
        ts_run = list(ts[skip:])

        if is_first or prev_tail is None:
            lat = torch.randn((1, pipe.unet.config.in_channels, chunk, lat_h, lat_h),
                              generator=gen, dtype=torch.float32)
        elif mode == "repeat-pad":
            reps = chunk - prev_tail.shape[2]
            lat = torch.cat([prev_tail, prev_tail[:, :, -1:].repeat(1, 1, reps, 1, 1)], dim=2)
            noise = torch.randn(lat.shape, generator=gen, dtype=torch.float32)
            # CRITICAL: noise to the timestep the solver will actually start from.
            # The old code noised to `carry_noise` (e.g. 500) while skip was derived from
            # `t <= carry_noise` (=3 -> t=400), a one-step mismatch. That mismatch is what
            # produced the +14%/chunk latent-std drift (whitening); matching them keeps std
            # flat (0.6207 -> 0.6139 -> 0.6310 in the matched 3-chunk run).
            if noise_t == "match":
                tt = torch.as_tensor([float(ts[skip])])
            else:
                tt = torch.as_tensor([float(noise_t)])
            lat = sched.add_noise(lat, noise, tt)
        else:  # independent
            keep = prev_tail.shape[2]
            noise_c = torch.randn(prev_tail.shape, generator=gen, dtype=torch.float32)
            tt = torch.as_tensor([float(ts[skip])])   # noise carried frames to the relay t
            carried = sched.add_noise(prev_tail, noise_c, tt)
            new = torch.randn((1, pipe.unet.config.in_channels, chunk - keep, lat_h, lat_h),
                              generator=gen, dtype=torch.float32)
            lat = torch.cat([carried, new], dim=2)

        lat = lat * sched.init_noise_sigma

        with torch.no_grad():
            for t in ts_run:
                mi = torch.cat([lat] * 2)
                mi = sched.scale_model_input(mi, t)
                npr = pipe.unet(mi, t, encoder_hidden_states=embeds).sample
                nu, nt = npr.chunk(2)
                npr = nu + GUIDANCE * (nt - nu)
                lat = sched.step(npr, t, lat,
                                 **pipe.prepare_extra_step_kwargs(gen, 0.0)).prev_sample

        with torch.no_grad():
            b, c, T, h, w = lat.shape
            flat = (lat / pipe.vae.config.scaling_factor).permute(0, 2, 1, 3, 4).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat, return_dict=False)[0]
            arr = (((dec / 2 + 0.5).clamp(0, 1)).reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                   .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype(np.uint8)

        st = [frame_stats(arr[i]) for i in range(len(arr))]
        per_chunk.append(dict(chunk=ci, skip=skip, steps=len(ts_run),
                              lap=float(np.mean([s["lap"] for s in st])),
                              white=float(np.mean([s["white"] for s in st])),
                              blown=float(np.mean([s["blown"] for s in st])),
                              green=int(np.sum([s["green_px"] for s in st])),
                              mean=float(np.mean([s["mean"] for s in st])),
                              lat_std=float(lat.std())))

        keep = max(1, overlap)
        if ci == 0:
            all_frames.extend(list(arr))
        else:
            new = arr[keep:]
            for j in range(min(keep, len(new))):
                w_ = (j + 1) / (keep + 1)
                all_frames[len(all_frames) - keep + j] = (
                    all_frames[len(all_frames) - keep + j].astype(np.float32) * (1 - w_)
                    + new[j].astype(np.float32) * w_).round().astype(np.uint8)
            all_frames.extend(list(new))
        prev_tail = lat[:, :, -keep:].clone()

    return per_chunk, all_frames


def report(name, per_chunk, frames, chunk, overlap):
    print(f"\n--- {name} ---")
    print(f"{'chunk':>6} {'steps':>6} {'lat std':>9} {'Laplacian var':>14} "
          f"{'white%':>8} {'blown%':>8} {'green_px':>9}")
    for s in per_chunk:
        print(f"{s['chunk']:>6} {s['steps']:>6} {s['lat_std']:>9.4f} {s['lap']:>14.1f} "
              f"{s['white']:>8.2f} {s['blown']:>8.2f} {s['green']:>9}")
    laps = [s["lap"] for s in per_chunk]
    drop = (laps[0] - laps[-1]) / laps[0] * 100 if laps[0] else 0
    print(f"  detail decay chunk0 -> chunk{len(laps)-1}: {laps[0]:.1f} -> {laps[-1]:.1f} "
          f"({drop:+.1f}%)")

    fr = [f.astype(np.float32) for f in frames]
    mads = [float(np.abs(fr[i + 1] - fr[i]).mean()) for i in range(len(fr) - 1)]
    stride = chunk - overlap
    bounds = set(range(stride, len(fr) - 1, stride))
    intra = [mads[i] for i in range(len(mads)) if i not in bounds]
    atb = [mads[i] for i in range(len(mads)) if i in bounds]
    print(f"  temporal: overall MAD {np.mean(mads):.2f} | intra {np.mean(intra):.2f} | "
          f"seam {np.mean(atb):.2f}")
    return dict(lap0=laps[0], lapN=laps[-1], drop=drop, mad=float(np.mean(mads)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--relay-steps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt", default=ANIME_PROMPT)
    ap.add_argument("--neg", default=ANIME_NEG)
    ap.add_argument("--modes", default="repeat-pad",
                    help="comma list of init modes to run: repeat-pad,independent")
    ap.add_argument("--noise-t", default="match",
                    help="'match' = add_noise at the schedule's actual relay start ts[skip] "
                         "(correct); or a number to reproduce the old mismatch")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    print(f"[cfg] res={a.res} chunk={a.chunk} overlap={a.overlap} chunks={a.chunks} "
          f"steps={a.steps} relay_steps={a.relay_steps}")

    pipe = build("dpm")
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, "cpu", 1, True, negative_prompt=a.neg)
    embeds = torch.cat([ne, pe])

    results = {}
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    for mode in modes:
        t0 = time.time()
        pc, frames = run(pipe, embeds, a.seed, a.res, a.chunk, a.overlap, a.chunks,
                         a.steps, a.relay_steps, 400, mode, a.prompt, a.neg,
                         noise_t=a.noise_t)
        print(f"\n[run] mode={mode} noise_t={a.noise_t} took {time.time()-t0:.0f}s")
        results[mode] = report(f"INIT = {mode} (noise_t={a.noise_t})", pc, frames,
                               a.chunk, a.overlap)

    print(f"\n{'='*72}")
    print("VERDICT")
    for mode, r in results.items():
        print(f"  {mode:>12}: detail {r['lap0']:.1f} -> {r['lapN']:.1f} "
              f"({r['drop']:+.1f}%)   overall MAD {r['mad']:.2f}")
    if "repeat-pad" in results:
        ra = results["repeat-pad"]
        print(f"\n  repeat-pad detail decay {ra['drop']:+.1f}% "
              f"-> {'accumulated smoothing' if ra['drop'] > 50 else 'one-time drop, then plateau'}")


if __name__ == "__main__":
    main()
