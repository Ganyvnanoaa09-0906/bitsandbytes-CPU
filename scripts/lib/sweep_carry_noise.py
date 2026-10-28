# -*- coding: utf-8 -*-
"""Sweep carry-noise for the sliding-window generator in a single model load.

Fixed : DPM-Solver++(2M), Counterfeit-V3 base, motion adapter v1-5, prompt/seed, steps.
Varies: --carry-noises (comma list)
Reports per value: boundary/intra ratio, intra-chunk MAD, at-boundary MAD,
histogram drift, total time. Also compares drift growth across chunk counts to test
whether appearance drift is linear in the number of chunks.

One model load serves every configuration (loading 3.3GB of weights per config would
dominate runtime otherwise).

Usage:
  python sweep_carry_noise.py --res 256 --chunk 10 --overlap 2 --chunks 3 --steps 6 \
      --carry-noises 300,400,500,600,700
"""
import os, sys
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
import sys
import time

import numpy
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from anime_adiff import build, ANIME_PROMPT, ANIME_NEG, OUT_DIR  # noqa: E402
import temporal_metrics

GUIDANCE = 7.5


def generate(pipe, embeds, gen_state, res, chunk, overlap, chunks, steps, carry_noise,
             prompt, neg):
    """Run one sliding-window generation. Returns (frames_uint8_list, elapsed_s, log)."""
    lat_h = res // pipe.vae_scale_factor
    sched = pipe.scheduler
    gen = torch.Generator().manual_seed(gen_state["seed"])
    if gen_state.get("warm") is not None:
        # keep the noise stream deterministic across sweep entries
        gen.set_state(gen_state["warm"])

    all_frames = []
    prev_tail = None
    t0 = time.time()
    log = []

    for ci in range(chunks):
        is_first = ci == 0
        sched.set_timesteps(steps)
        ts = sched.timesteps
        skip = 0
        if not is_first:
            below = [i for i, t in enumerate(ts) if float(t) <= carry_noise]
            skip = below[0] if below else max(0, len(ts) - 1)
            sched._step_index = skip
            sched.model_outputs = [None] * sched.config.solver_order
            sched._lower_order_nums = 0
        ts_run = ts[skip:]

        if is_first or prev_tail is None:
            lat = torch.randn((1, pipe.unet.config.in_channels, chunk, lat_h, lat_h),
                              generator=gen, dtype=torch.float32)
        else:
            tail = prev_tail
            reps = chunk - tail.shape[2]
            if reps > 0:
                lat = torch.cat([tail, tail[:, :, -1:].repeat(1, 1, reps, 1, 1)], dim=2)
            else:
                lat = tail[:, :, :chunk]
            noise = torch.randn(lat.shape, generator=gen, dtype=torch.float32)
            # CRITICAL: noise to the timestep the solver actually starts from (ts[skip]).
            # Using carry_noise directly mismatches it by one schedule step (carry_noise=500
            # while skip lands on t=400), which makes the solver misjudge the starting noise
            # level and add net energy per chunk -> whitening drift (+32.4% latent std over
            # 4 chunks vs +1.8% when matched).
            tt = torch.as_tensor([float(ts[skip])])
            lat = sched.add_noise(lat, noise, tt)

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
            flat = lat.permute(0, 2, 1, 3, 4).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat / pipe.vae.config.scaling_factor, return_dict=False)[0]
            dec = (dec / 2 + 0.5).clamp(0, 1)
            arr = (dec.reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                      .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype(np.uint8)

        keep = max(1, overlap)
        prev_tail = lat[:, :, -keep:].clone()
        if is_first:
            all_frames.extend(list(arr))
        else:
            new = arr[keep:]
            for j in range(min(keep, len(new))):
                w_ = (j + 1) / (keep + 1)
                all_frames[-keep + j] = (all_frames[-keep + j].astype(np.float32) * (1 - w_)
                                         + new[j].astype(np.float32) * w_).round().astype(np.uint8)
            all_frames.extend(list(new))
        log.append(f"chunk{ci} skip={skip} steps={len(ts_run)}")

    return all_frames, time.time() - t0, log


def measure(frames, chunk, overlap):
    fr = [f.astype(np.float32) for f in frames]
    mads = [float(np.abs(fr[i + 1] - fr[i]).mean()) for i in range(len(fr) - 1)]
    hists = [float(tm.hist_l1(fr[i], fr[i + 1])) for i in range(len(fr) - 1)]
    stride = chunk - overlap
    bounds = set(range(stride, len(fr) - 1, stride))
    intra = [mads[i] for i in range(len(mads)) if i not in bounds]
    atb = [mads[i] for i in range(len(mads)) if i in bounds]
    return dict(
        mad=float(np.mean(mads)),
        intra=float(np.mean(intra)) if intra else None,
        boundary=float(np.mean(atb)) if atb else None,
        ratio=(float(np.mean(atb)) / float(np.mean(intra))) if (intra and atb) else None,
        hist=float(np.mean(hists)),
        mad_first=mads[0],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--carry-noises", default="300,400,500,600,700")
    ap.add_argument("--chunk-count-check", default="2,3",
                    help="chunk counts to compare for drift-linearity (uses best carry-noise)")
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--prompt", default=ANIME_PROMPT)
    ap.add_argument("--neg", default=ANIME_NEG)
    ap.add_argument("--save", action="store_true", help="save PNGs for each config")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    noises = [float(x) for x in a.carry_noises.split(",") if x.strip()]

    print(f"[cfg] res={a.res} chunk={a.chunk} overlap={a.overlap} chunks={a.chunks} "
          f"steps={a.steps} solver=DPM-Solver++(2M) base=counterfeit-v3", flush=True)
    print(f"[cfg] carry-noise sweep = {noises}", flush=True)

    pipe = build("dpm")
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, "cpu", 1, True, negative_prompt=a.neg)
    embeds = torch.cat([ne, pe])

    # deterministic noise stream: capture the generator state after chunk 0 so every
    # sweep entry starts its carried chunks from the same noise.
    gprobe = torch.Generator().manual_seed(a.seed)
    torch.randn((1, 4, a.chunk, a.res // pipe.vae_scale_factor, a.res // pipe.vae_scale_factor),
                generator=gprobe)
    warm = gprobe.get_state().clone()
    gen_state = {"seed": a.seed, "warm": warm}

    results = []
    os.makedirs(OUT_DIR, exist_ok=True)

    for cn in noises:
        print(f"\n{'='*60}\n[run] carry-noise = {cn}\n{'='*60}", flush=True)
        frames, dt, log = generate(pipe, embeds, dict(gen_state), a.res, a.chunk,
                                   a.overlap, a.chunks, a.steps, cn, a.prompt, a.neg)
        m = measure(frames, a.chunk, a.overlap)
        m.update(carry_noise=cn, chunks=a.chunks, seconds=dt, frames=len(frames))
        results.append(m)
        print(f"[result] cn={cn} mad={m['mad']:.2f} intra={m['intra']:.2f} "
              f"boundary={m['boundary']:.2f} ratio={m['ratio']:.2f} "
              f"hist={m['hist']:.4f} time={dt:.1f}s", flush=True)
        print(f"         {'; '.join(log)}", flush=True)
        if a.save:
            tag = f"{a.tag}_{a.res}_cn{int(cn)}"
            fdir = os.path.join(OUT_DIR, tag + "_frames")
            os.makedirs(fdir, exist_ok=True)
            for i, f in enumerate(frames):
                Image.fromarray(f).convert("RGB").save(
                    os.path.join(fdir, f"f{i:03d}.png"))
            print(f"         frames -> {fdir}", flush=True)

    # ---- best carry-noise by boundary/intra ratio ----
    valid = [r for r in results if r["ratio"] is not None]
    best = min(valid, key=lambda r: r["ratio"]) if valid else None
    baseline_intra = results[0]["intra"] if results else None

    print(f"\n\n{'#'*60}\n# SUMMARY (res={a.res}, chunk={a.chunk}, overlap={a.overlap}, "
          f"chunks={a.chunks}, steps={a.steps})\n{'#'*60}")
    print(f"{'carry-noise':>12} {'intra MAD':>10} {'boundary':>10} {'ratio':>7} "
          f"{'hist':>8} {'time(s)':>9}")
    for r in results:
        print(f"{r['carry_noise']:>12.0f} {r['intra']:>10.2f} {r['boundary']:>10.2f} "
              f"{r['ratio']:>7.2f} {r['hist']:>8.4f} {r['seconds']:>9.1f}")
    if best:
        print(f"\nBEST carry-noise = {best['carry_noise']:.0f} "
              f"(ratio={best['ratio']:.2f}, intra={best['intra']:.2f})")
        if baseline_intra is not None:
            drop = (baseline_intra - best["intra"]) / baseline_intra * 100
            print(f"  intra MAD change vs first config: {drop:+.1f}% "
                  f"({'OK, <10% drop' if drop <= 10 else 'WARNING: intra worsened >10%'})")

    # ---- drift linearity across chunk counts ----
    counts = [int(x) for x in a.chunk_count_check.split(",") if x.strip()]
    if best and len(counts) > 1:
        print(f"\n--- drift vs chunk count (carry-noise={best['carry_noise']:.0f}) ---")
        lin = []
        for cc in counts:
            frames, dt, _ = generate(pipe, embeds, dict(gen_state), a.res, a.chunk,
                                     a.overlap, cc, a.steps, best["carry_noise"],
                                     a.prompt, a.neg)
            m = measure(frames, a.chunk, a.overlap)
            lin.append(dict(chunks=cc, frames=len(frames), hist=m["hist"], mad=m["mad"],
                            seconds=dt))
            print(f"  chunks={cc} frames={len(frames)} hist_drift={m['hist']:.4f} "
                  f"mad={m['mad']:.2f} time={dt:.1f}s", flush=True)
        if len(lin) == 2:
            (c0, c1) = lin
            f0, f1 = c0["frames"], c1["frames"]
            exp = c0["hist"] * (f1 / f0)
            print(f"  linear expectation for {f1} frames: {exp:.4f}")
            print(f"  actual                            : {c1['hist']:.4f}")
            verdict = ("LINEAR (needs global style anchor)" if c1["hist"] > exp * 1.15
                       else "SUB-LINEAR (drift self-limits; no anchor needed)")
            print(f"  -> {verdict}")

    out = os.path.join(OUT_DIR, f"{a.tag}_{a.res}_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(config=dict(res=a.res, chunk=a.chunk, overlap=a.overlap,
                                   chunks=a.chunks, steps=a.steps, noises=noises),
                       results=results, best=best), f, indent=2)
    print(f"\n[save] {out}", flush=True)


if __name__ == "__main__":
    main()
