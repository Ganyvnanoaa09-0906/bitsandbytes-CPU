# -*- coding: utf-8 -*-
"""Fast root-cause check for the carried-latent energy drift.

Hypothesis: carried chunks leave residual noise because they stop at the schedule's last
timestep (t=200 for a 5-step schedule) instead of being driven all the way down; combined
with an emptied multi-step solver history, each chunk adds a bit of energy -> latent std
grows ~15% per chunk -> decoded highlights blow out (white% x5-8 per chunk).

Test at 256 / 2 chunks (cheap): compare the carried chunk running
  (a) the tail as-is               (current behaviour)
  (b) the tail PLUS a final step to t=0  (fix candidate)

Decision rule: if (b) makes chunk1 final std within 3% of chunk0, root cause confirmed.

Usage:
  python probe_drift_fix.py --res 256 --chunk 10 --overlap 2 --steps 5 --carry-noise 500
"""
import os, sys
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

import torch
import numpy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.anime_adiff import build, ANIME_PROMPT, ANIME_NEG, OUT_DIR

GUIDANCE = 7.5


def tstats(t):
    f = t.detach().float().flatten()
    return dict(mean=float(f.mean()), std=float(f.std()),
                absmax=float(f.abs().max()), p99=float(f.abs().quantile(0.99)))


def run_full_schedule(pipe, embeds, gen_seed, res, chunk, overlap, chunks, steps,
                      t_start, prompt, neg, label):
    """Relay with the scheduler's NATIVE add_noise, running the full remaining schedule.

    Difference from the earlier relay variants: the carried latent is noised with
    scheduler.add_noise at a real schedule timestep, so the relay start IS on the
    forward-diffusion manifold (the earlier `carried*r + noise*(1-r)` mixes were
    off-manifold, which is why more relay steps made drift worse, +32% -> +196%).

    With T_start=799 on a 5-step schedule [999,799,599,400,200] the relay chunk runs
    [799,599,400,200] -- identical to chunk 0 except for the starting noise level, which
    isolates "start noise level" as the single variable.
    """
    lat_h = res // pipe.vae_scale_factor
    sched = pipe.scheduler
    gen = torch.Generator().manual_seed(gen_seed)
    torch.randn((1, 4, chunk, lat_h, lat_h), generator=gen)

    prev_tail = None
    stats = []
    all_frames = []

    for ci in range(chunks):
        is_first = ci == 0
        sched.set_timesteps(steps)
        ts = sched.timesteps

        if is_first:
            skip = 0
        else:
            idx = sched.index_for_timestep(torch.tensor(int(t_start)), ts)
            skip = int(idx)
        sched._step_index = skip
        sched.model_outputs = [None] * sched.config.solver_order
        sched._lower_order_nums = 0
        ts_run = list(ts[skip:])

        if is_first or prev_tail is None:
            lat = torch.randn((1, pipe.unet.config.in_channels, chunk, lat_h, lat_h),
                              generator=gen, dtype=torch.float32)
        else:
            reps = chunk - prev_tail.shape[2]
            if reps > 0:
                lat = torch.cat([prev_tail, prev_tail[:, :, -1:].repeat(1, 1, reps, 1, 1)], dim=2)
            else:
                lat = prev_tail[:, :, :chunk]
            noise = torch.randn(lat.shape, generator=gen, dtype=torch.float32)
            # NATIVE scheduler noising at a real timestep -> stays on-manifold
            lat = sched.add_noise(lat, noise, torch.as_tensor([int(t_start)]))

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

        s = tstats(lat)
        with torch.no_grad():
            b, c, T, h, w = lat.shape
            flat = (lat / pipe.vae.config.scaling_factor).permute(0, 2, 1, 3, 4).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat, return_dict=False)[0]
            raw_over = float((dec / 2 + 0.5 > 1.0).float().mean())
            arr = (((dec / 2 + 0.5).clamp(0, 1)).reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                   .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype("uint8")
        s.update(chunk=ci, steps_run=len(ts_run), skip=skip, t_start=int(t_start),
                 white_pct=float(100 * (arr > 245).mean()), raw_over=raw_over)

        if ci == 0:
            all_frames.extend(list(arr))
        else:
            keep = max(1, overlap)
            new = arr[keep:]
            for j in range(min(keep, len(new))):
                w_ = (j + 1) / (keep + 1)
                all_frames[len(all_frames) - keep + j] = (
                    all_frames[len(all_frames) - keep + j].astype(np.float32) * (1 - w_)
                    + new[j].astype(np.float32) * w_).round().astype(np.uint8)
            all_frames.extend(list(new))

        stats.append(s)
        keep = max(1, overlap)
        prev_tail = lat[:, :, -keep:].clone()

    print(f"\n--- {label} ---")
    print(f"{'chunk':>6} {'skip':>5} {'steps':>6} {'final std':>10} {'absmax':>9} "
          f"{'|p99|':>7} {'raw>1%':>8} {'white%':>8}")
    for s in stats:
        print(f"{s['chunk']:>6} {s['skip']:>5} {s['steps_run']:>6} {s['std']:>10.4f} "
              f"{s['absmax']:>9.3f} {s['p99']:>7.3f} {s['raw_over']*100:>8.3f} "
              f"{s['white_pct']:>8.2f}")
    return stats, all_frames


def run_independent(pipe, embeds, gen_seed, res, chunk, overlap, chunks, steps,
                    prompt, neg, label, same_seed=False):
    """Independent chunk sampling + pixel-space overlap blending.

    No latent carry, no add_noise, no latent blending: every chunk is a plain full-schedule
    sample from pure noise (t=999). Chunks are therefore each on-manifold -- the property
    the latent-relay variants violated (their inputs were off the forward-diffusion
    distribution, which is why 3 relay steps made drift WORSE, +196% vs +32%).

    Overlap frames are blended in pixel space after VAE decode.
    same_seed=False gives each chunk its own seed (the honest test of chunk-to-chunk
    content agreement). same_seed=True would make chunks near-identical and would
    flatter the seam metric without proving anything.
    """
    lat_h = res // pipe.vae_scale_factor
    sched = pipe.scheduler
    all_frames = []
    stats = []
    seam_idx = []

    for ci in range(chunks):
        sched.set_timesteps(steps)
        sched._step_index = 0
        sched.model_outputs = [None] * sched.config.solver_order
        sched._lower_order_nums = 0
        ts_run = list(sched.timesteps)

        seed = gen_seed if same_seed else gen_seed + ci * 1000
        gen = torch.Generator().manual_seed(seed)

        lat = torch.randn((1, pipe.unet.config.in_channels, chunk, lat_h, lat_h),
                          generator=gen, dtype=torch.float32)
        s_init = tstats(lat)
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

        s_fin = tstats(lat)
        lat_in = lat / pipe.vae.config.scaling_factor
        with torch.no_grad():
            b, c, T, h, w = lat.shape
            flat = lat_in.permute(0, 2, 1, 3, 4).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat, return_dict=False)[0]
            raw_over = float((dec / 2 + 0.5 > 1.0).float().mean())
            arr = (((dec / 2 + 0.5).clamp(0, 1)).reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                   .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype("uint8")

        white = float(100 * (arr > 245).mean())
        s_fin.update(chunk=ci, steps_run=len(ts_run), seed=seed,
                     init_std=s_init["std"], white_pct=white, raw_over=raw_over,
                     absmax=s_fin["absmax"], p99=s_fin["p99"])
        stats.append(s_fin)

        if ci == 0:
            all_frames.extend(list(arr))
        else:
            keep = max(1, overlap)
            new = arr[keep:]
            # pixel-space linear blend across the overlap, then record the seam index
            for j in range(min(keep, len(new))):
                w_ = (j + 1) / (keep + 1)
                all_frames[len(all_frames) - keep + j] = (
                    all_frames[len(all_frames) - keep + j].astype(np.float32) * (1 - w_)
                    + new[j].astype(np.float32) * w_).round().astype(np.uint8)
            seam_idx.append(len(all_frames))
            all_frames.extend(list(new))

    print(f"\n--- {label} ---")
    print(f"{'chunk':>6} {'steps':>6} {'init std':>9} {'final std':>10} {'absmax':>9} "
          f"{'|p99|':>7} {'raw>1%':>8} {'white%':>8}")
    for s in stats:
        print(f"{s['chunk']:>6} {s['steps_run']:>6} {s['init_std']:>9.4f} {s['std']:>10.4f} "
              f"{s['absmax']:>9.3f} {s['p99']:>7.3f} {s['raw_over']*100:>8.3f} "
              f"{s['white_pct']:>8.2f}")
    return stats, all_frames, seam_idx


def run(pipe, embeds, gen_seed, res, chunk, overlap, chunks, steps, carry_noise,
        blend_ratio, prompt, neg, label, norm_mode="none", relay_steps=None):
    """Generate with explicit latent blending for carried chunks.

    blend_ratio=None -> legacy behaviour: scheduler.add_noise(carried, t<=carry_noise).
    blend_ratio=r    -> lat = carried*r + noise*(1-r), bypassing the scheduler sigma maths.

    norm_mode controls WHERE a std normalisation is applied (None/off by default):
      "none"   : no normalisation.
      "input"  : normalise the carried+noised latent BEFORE denoising, so the carried chunk
                 starts at the same energy scale as chunk 0's start. This is the
                 informative test: whatever std the chunk then produces is natural.
      "output" : normalise the latent AFTER denoising, before carrying it forward. NOTE:
                 this forces every chunk's std to equal chunk 0's, so the measured drift is
                 ~0 BY CONSTRUCTION -- it validates nothing, it only shows the plumbing.
    """
    lat_h = res // pipe.vae_scale_factor
    sched = pipe.scheduler
    gen = torch.Generator().manual_seed(gen_seed)
    torch.randn((1, 4, chunk, lat_h, lat_h), generator=gen)

    prev_tail = None
    stats = []
    all_frames = []
    std0 = None

    for ci in range(chunks):
        is_first = ci == 0
        sched.set_timesteps(steps)
        ts = sched.timesteps
        skip = 0
        if not is_first:
            if relay_steps is not None:
                # explicit: run the last `relay_steps` entries of the full schedule
                skip = max(0, len(ts) - relay_steps)
            else:
                below = [i for i, t in enumerate(ts) if float(t) <= carry_noise]
                skip = below[0] if below else max(0, len(ts) - 1)
            sched._step_index = skip
            sched.model_outputs = [None] * sched.config.solver_order
            sched._lower_order_nums = 0
        ts_run = list(ts[skip:])

        if is_first or prev_tail is None:
            lat = torch.randn((1, pipe.unet.config.in_channels, chunk, lat_h, lat_h),
                              generator=gen, dtype=torch.float32)
        else:
            reps = chunk - prev_tail.shape[2]
            if reps > 0:
                lat = torch.cat([prev_tail, prev_tail[:, :, -1:].repeat(1, 1, reps, 1, 1)], dim=2)
            else:
                lat = prev_tail[:, :, :chunk]
            noise = torch.randn(lat.shape, generator=gen, dtype=torch.float32)
            if blend_ratio is None:
                # noise to the schedule's actual start (ts[skip]); using carry_noise directly
                # mismatched by one step and caused +32.4% latent-std drift over 4 chunks.
                tt = torch.as_tensor([float(ts[skip])])
                lat = sched.add_noise(lat, noise, tt)
            else:
                lat = lat * blend_ratio + noise * (1.0 - blend_ratio)

        # --- INPUT normalisation: match the carried start's scale to chunk 0's start ---
        if (not is_first) and norm_mode == "input" and std0 is not None:
            lat = lat * (std0 / lat.std())

        lat = lat * sched.init_noise_sigma
        if is_first:
            std0 = float(lat.std())

        with torch.no_grad():
            for t in ts_run:
                mi = torch.cat([lat] * 2)
                mi = sched.scale_model_input(mi, t)
                npr = pipe.unet(mi, t, encoder_hidden_states=embeds).sample
                nu, nt = npr.chunk(2)
                npr = nu + GUIDANCE * (nt - nu)
                lat = sched.step(npr, t, lat,
                                 **pipe.prepare_extra_step_kwargs(gen, 0.0)).prev_sample

        if is_first:
            std0_out = float(lat.std())

        # --- OUTPUT normalisation: force the carried latent back to chunk 0's energy ---
        norm_delta = None
        if (not is_first) and norm_mode == "output":
            before_std = float(lat.std())
            scale = std0_out / before_std
            lat = lat * scale
            norm_delta = dict(scale=scale, std_before=before_std,
                              std_after=float(lat.std()), chunk=ci)
            # normalisation is a pure scalar multiply: content is provably unchanged
            print(f"    [norm] chunk{ci} scale={scale:.4f}  "
                  f"std {before_std:.4f} -> {float(lat.std()):.4f}", flush=True)

        s = tstats(lat)
        s["chunk"] = ci
        s["steps_run"] = len(ts_run)
        stats.append(s)

        with torch.no_grad():
            b, c, T, h, w = lat.shape
            flat = (lat / pipe.vae.config.scaling_factor).permute(0, 2, 1, 3, 4).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat, return_dict=False)[0]
            arr = (((dec / 2 + 0.5).clamp(0, 1)).reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                   .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype("uint8")
        s["white_pct"] = float(100 * (arr > 245).mean())

        if ci == 0:
            all_frames.extend(list(arr))
        else:
            keep = max(1, overlap)
            new = arr[keep:]
            for j in range(min(keep, len(new))):
                w_ = (j + 1) / (keep + 1)
                all_frames[len(all_frames) - keep + j] = (
                    all_frames[len(all_frames) - keep + j].astype(np.float32) * (1 - w_)
                    + new[j].astype(np.float32) * w_).round().astype(np.uint8)
            all_frames.extend(list(new))

        keep = max(1, overlap)
        prev_tail = lat[:, :, -keep:].clone()

    print(f"\n--- {label} ---")
    print(f"{'chunk':>6} {'steps':>6} {'std':>9} {'absmax':>9} {'|p99|':>7} {'white%':>8} "
          f"{'std vs c0':>10}")
    base = stats[0]["std"]
    for s in stats:
        d = (s["std"] - base) / base * 100
        print(f"{s['chunk']:>6} {s['steps_run']:>6} {s['std']:>9.4f} {s['absmax']:>9.3f} "
              f"{s['p99']:>7.3f} {s['white_pct']:>8.2f} {d:>9.2f}%")
    last = stats[-1]
    d = (last["std"] - base) / base * 100
    print(f"  LAST chunk vs chunk0: {d:+.2f}%")
    return stats, all_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--carry-noise", type=float, default=500.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ratios", default="0.8",
                    help="comma list of explicit blend ratios; 'legacy' = use scheduler.add_noise")
    ap.add_argument("--skip-legacy", action="store_true",
                    help="do not run the legacy add_noise baseline")
    ap.add_argument("--norm", default="none,input,output",
                    help="comma list of normalisation modes: none|input|output")
    ap.add_argument("--relay-steps", default=None,
                    help="comma list of relay chunk step counts (e.g. '2,3'); "
                         "overrides carry-noise's implicit skip")
    ap.add_argument("--mode", default="relay",
                    choices=["relay", "independent", "full_schedule"],
                    help="relay = latent carry (known to diverge); "
                         "independent = per-chunk full sampling + pixel overlap blend; "
                         "full_schedule = relay with native scheduler.add_noise at a real "
                         "timestep, running the full remaining schedule")
    ap.add_argument("--t-starts", default="799",
                    help="comma list of T_start timesteps for --mode full_schedule")
    ap.add_argument("--same-seed", action="store_true",
                    help="independent mode: use one seed for all chunks (near-identical "
                         "chunks; flatters the seam metric, for reference only)")
    ap.add_argument("--prompt", default=ANIME_PROMPT)
    ap.add_argument("--neg", default=ANIME_NEG)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    print(f"[cfg] res={a.res} chunk={a.chunk} chunks={a.chunks} steps={a.steps} cn={a.carry_noise}")

    pipe = build("dpm")
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, "cpu", 1, True, negative_prompt=a.neg)
    embeds = torch.cat([ne, pe])

    # NOTE: the legacy baseline must NOT run for --mode independent; it used to run
    # unconditionally and wasted a full ~8 minute generation before the branch was taken.
    base = None
    if a.mode != "independent" and not a.skip_legacy:
        base = run(pipe, embeds, a.seed, a.res, a.chunk, a.overlap, a.chunks, a.steps,
                   a.carry_noise, None, a.prompt, a.neg,
                   "BASELINE (scheduler.add_noise, legacy)")

    ratios = [None if x.strip() == "legacy" else float(x)
              for x in a.ratios.split(",") if x.strip()]
    norms = [x.strip() for x in a.norm.split(",") if x.strip()]

    fixed = {}
    if a.mode == "full_schedule":
        t_starts = [int(x) for x in a.t_starts.split(",") if x.strip()]
        for ts_ in t_starts:
            st, frames = run_full_schedule(
                pipe, embeds, a.seed, a.res, a.chunk, a.overlap, a.chunks, a.steps,
                ts_, a.prompt, a.neg,
                f"FULL SCHEDULE relay, native add_noise T_start={ts_}")

            w = [s["white_pct"] for s in st]
            fr = [f.astype(np.float32) for f in frames]
            mads = [float(np.abs(fr[i + 1] - fr[i]).mean()) for i in range(len(fr) - 1)]
            stride = a.chunk - a.overlap
            bounds = set(range(stride, len(fr) - 1, stride))
            intra = [mads[i] for i in range(len(mads)) if i not in bounds]
            atb = [mads[i] for i in range(len(mads)) if i in bounds]
            ii = float(np.mean(intra)) if intra else float("nan")
            bb = float(np.mean(atb)) if atb else float("nan")
            overall = float(np.mean(mads))

            print(f"\n  [white% curve] " + " ".join(f"{x:.2f}" for x in w))
            rising = all(w[i + 1] >= w[i] - 0.05 for i in range(len(w) - 1))
            print(f"  monotone-rising={rising}   spread={max(w)-min(w):.2f}")
            print(f"  [temporal] overall MAD {overall:.2f} | intra {ii:.2f} | seam {bb:.2f} | "
                  f"ratio {bb/ii if intra else float('nan'):.2f}")
            print(f"  [verdict] white%stable={not rising}  MAD<10={overall < 10}  "
                  f"MAD<15={overall < 15}  -> "
                  f"{'CANDIDATE 1 WORKS' if (not rising and overall < 10) else ('partial: temporal ok, whitening persists' if (not rising) else 'whitening persists -> go to candidate 2')}")
            fixed[f"T_start={ts_}"] = st
        return

    if a.mode == "independent":
        st, frames, seams = run_independent(
            pipe, embeds, a.seed, a.res, a.chunk, a.overlap, a.chunks, a.steps,
            a.prompt, a.neg,
            f"INDEPENDENT chunks (per-chunk full {a.steps}-step sampling, pixel blend)",
            same_seed=a.same_seed)

        # ---- white% stability ----
        w = [s["white_pct"] for s in st]
        print(f"\n[criterion 1] white% across chunks: " +
              " ".join(f"{x:.2f}" for x in w))
        rising = all(w[i + 1] >= w[i] - 0.5 for i in range(len(w) - 1))
        spread = max(w) - min(w)
        print(f"  monotone-rising={rising}  spread={spread:.2f}  -> "
              f"{'FAIL (whitening accumulates)' if rising and spread > 2 else 'PASS (stable)'}")

        # ---- seam vs intra MAD ----
        fr = [f.astype(np.float32) for f in frames]
        mads = [float(np.abs(fr[i + 1] - fr[i]).mean()) for i in range(len(fr) - 1)]
        stride = a.chunk - a.overlap
        bounds = set(range(stride, len(fr) - 1, stride))
        intra = [mads[i] for i in range(len(mads)) if i not in bounds]
        atb = [mads[i] for i in range(len(mads)) if i in bounds]
        print(f"\n[criterion 2] frames={len(fr)}  per-transition MAD:")
        print("   " + " ".join(f"{m:.1f}" for m in mads))
        ii = float(np.mean(intra)) if intra else float("nan")
        bb = float(np.mean(atb)) if atb else float("nan")
        print(f"  intra-chunk MAD  : {ii:.2f}  (n={len(intra)})")
        print(f"  SEAM MAD         : {bb:.2f}  (n={len(atb)}, indices={sorted(bounds)})")
        if intra and atb:
            r = bb / ii
            print(f"  seam/intra ratio : {r:.2f}  -> "
                  f"{'PASS (<1.5)' if r < 1.5 else ('MARGINAL' if r < 2 else 'FAIL (>2: chunks disagree)')}")
        print(f"\n  overall MAD: {np.mean(mads):.2f}")

        out = dict(mode="independent", res=a.res, chunks=a.chunks, steps=a.steps,
                   white=[s["white_pct"] for s in st],
                   intra=ii, seam=bb, ratio=(bb / ii if intra else None))
        import json
        with open(os.path.join(OUT_DIR, f"independent_{a.res}_{a.chunks}c.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        return

    if a.relay_steps:
        rsteps = [int(x) for x in a.relay_steps.split(",") if x.strip()]
        for rs in rsteps:
            for r in ratios:
                tag = f"relay={rs}step ratio={r}"
                fixed[tag] = run(pipe, embeds, a.seed, a.res, a.chunk, a.overlap, a.chunks,
                                 a.steps, a.carry_noise, r, a.prompt, a.neg,
                                 f"RELAY {rs}-STEP blend_ratio={r}", norm_mode="none",
                                 relay_steps=rs)
    else:
        for r in ratios:
            for nm in norms:
                tag = (f"ratio={r}" if r is not None else "legacy") + f" norm={nm}"
                fixed[tag] = run(pipe, embeds, a.seed, a.res, a.chunk, a.overlap, a.chunks,
                                 a.steps, a.carry_noise, r, a.prompt, a.neg,
                                 f"BLEND {tag}", norm_mode=nm)

    # ---- temporal analysis for every relay variant (MAD / seam / intra) ----
    print(f"\n{'='*90}")
    print("TEMPORAL ANALYSIS (block seam vs intra-chunk)")
    for tag, (st, frames) in fixed.items():
        if not frames:
            continue
        fr = [f.astype(np.float32) for f in frames]
        mads = [float(np.abs(fr[i + 1] - fr[i]).mean()) for i in range(len(fr) - 1)]
        stride = a.chunk - a.overlap
        bounds = set(range(stride, len(fr) - 1, stride))
        intra = [mads[i] for i in range(len(mads)) if i not in bounds]
        atb = [mads[i] for i in range(len(mads)) if i in bounds]
        ii = float(np.mean(intra)) if intra else float("nan")
        bb = float(np.mean(atb)) if atb else float("nan")
        print(f"  {tag}: frames={len(fr)} overall MAD={np.mean(mads):.2f}  "
              f"intra={ii:.2f}  seam={bb:.2f}  ratio={bb/ii if intra else float('nan'):.2f}")

    print(f"\n{'='*90}")
    print(f"VERDICT  ({a.chunks} chunks, res={a.res}, steps={a.steps}, cn={a.carry_noise})")
    print("  criterion: |std drift of LAST chunk vs chunk0| < 6%  (and white% must not rise)")
    print(f"{'variant':>28} {'c0 std':>9} {'last std':>9} {'drift':>9} {'white c0->last':>16} "
          f"{'verdict':>9}")
    print("-" * 90)
    for tag, (st, _frames) in fixed.items():
        b0 = st[0]["std"]
        last = st[-1]
        d = (last["std"] - b0) / b0 * 100
        w = [s["white_pct"] for s in st]
        ok = abs(d) < 6
        print(f"{tag:>28} {b0:>9.4f} {last['std']:>9.4f} {d:>8.2f}% "
              f"{w[0]:>6.2f}->{w[-1]:<6.2f} {'PASS' if ok else 'FAIL':>9}")
        print(f"{'':>28} per-chunk std: " + " ".join(f"{s['std']:.4f}" for s in st) +
              "   white%: " + " ".join(f"{s['white_pct']:.2f}" for s in st))


if __name__ == "__main__":
    main()
