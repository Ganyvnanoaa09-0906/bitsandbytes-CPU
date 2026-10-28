# -*- coding: utf-8 -*-
"""Probe the latent statistics of every chunk to locate the whitening/haze drift.

For each chunk this records, at three points:
  init   : the latent we start from (random noise for chunk 0; carried+noised for k>0)
  final  : the latent after the denoising loop (this is what gets carried forward)
  decoded: the decoded RGB frame stats

Whiten drift hypothesis: the carried latent's absolute scale grows chunk over chunk, so
each接力块 starts from an already "hot" latent, and VAE decode pushes highlights past 255.

Usage:
  python probe_latent.py --res 512 --chunk 10 --overlap 2 --chunks 4 --steps 5 --carry-noise 500
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
import re
import sys
import time

import numpy
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.anime_adiff import build, ANIME_PROMPT, ANIME_NEG


def tstats(t, name):
    f = t.detach().float().flatten()
    return dict(
        name=name,
        mean=float(f.mean()),
        std=float(f.std()),
        absmax=float(f.abs().max()),
        p99=float(f.abs().quantile(0.99)),
        p01=float(f.quantile(0.01)),
    )


def show(label, s):
    print(f"    {label:8s} mean={s['mean']:+.4f} std={s['std']:.4f} "
          f"absmax={s['absmax']:.3f} |p99|={s['p99']:.3f} p01={s['p01']:+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--carry-noise", type=float, default=500.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scaling-factor", type=float, default=None,
                    help="override VAE scaling factor (default: use vae.config)")
    ap.add_argument("--prompt", default=ANIME_PROMPT)
    ap.add_argument("--neg", default=ANIME_NEG)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    GUIDANCE = 7.5

    pipe = build("dpm")
    vae_sf = a.scaling_factor if a.scaling_factor is not None else pipe.vae.config.scaling_factor
    print(f"[cfg] res={a.res} chunk={a.chunk} overlap={a.overlap} chunks={a.chunks} "
          f"steps={a.steps} cn={a.carry_noise}")
    print(f"[cfg] vae scaling_factor = {vae_sf}  "
          f"shift_factor = {getattr(pipe.vae.config, 'shift_factor', None)}", flush=True)

    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, "cpu", 1, True, negative_prompt=a.neg)
    embeds = torch.cat([ne, pe])

    lat_h = a.res // pipe.vae_scale_factor
    sched = pipe.scheduler
    gen = torch.Generator().manual_seed(a.seed)

    prev_tail = None
    rows = []

    for ci in range(a.chunks):
        is_first = ci == 0
        sched.set_timesteps(a.steps)
        ts = sched.timesteps
        skip = 0
        if not is_first:
            below = [i for i, t in enumerate(ts) if float(t) <= a.carry_noise]
            skip = below[0] if below else max(0, len(ts) - 1)
            sched._step_index = skip
            sched.model_outputs = [None] * sched.config.solver_order
            sched._lower_order_nums = 0
        ts_run = ts[skip:]

        print(f"\n=== chunk {ci} (skip={skip}, steps={len(ts_run)}) ===")

        if is_first or prev_tail is None:
            lat = torch.randn((1, pipe.unet.config.in_channels, a.chunk, lat_h, lat_h),
                              generator=gen, dtype=torch.float32)
            show("init", tstats(lat, "init"))
        else:
            tail = prev_tail
            reps = a.chunk - tail.shape[2]
            if reps > 0:
                lat = torch.cat([tail, tail[:, :, -1:].repeat(1, 1, reps, 1, 1)], dim=2)
            else:
                lat = tail[:, :, :a.chunk]
            show("carried", tstats(lat, "carried"))
            noise = torch.randn(lat.shape, generator=gen, dtype=torch.float32)
            # noise to the schedule's actual start (ts[skip]), not carry_noise: the mismatch
            # was worth +32.4% latent-std drift over 4 chunks vs +1.8% when matched.
            tt = torch.as_tensor([float(ts[skip])])
            lat = sched.add_noise(lat, noise, tt)
            show("+noise", tstats(lat, "+noise"))

        lat = lat * sched.init_noise_sigma

        for t in ts_run:
            with torch.no_grad():
                mi = torch.cat([lat] * 2)
                mi = sched.scale_model_input(mi, t)
                npr = pipe.unet(mi, t, encoder_hidden_states=embeds).sample
                nu, nt = npr.chunk(2)
                npr = nu + GUIDANCE * (nt - nu)
                lat = sched.step(npr, t, lat,
                                 **pipe.prepare_extra_step_kwargs(gen, 0.0)).prev_sample

        fin = tstats(lat, "final")
        show("final", fin)

        # what the VAE actually receives
        lat_in = lat / vae_sf
        show("lat/vae_sf", tstats(lat_in, "lat_in"))

        with torch.no_grad():
            b, c, T, h, w = lat.shape
            flat = lat_in.permute(0, 2, 1, 3, 4).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat, return_dict=False)[0]
            dec = (dec / 2 + 0.5).clamp(0, 1)
            arr = (dec.reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                      .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype(np.uint8)

        # pre-clamp decode stats (to see how far past 1.0 the raw decode goes)
        with torch.no_grad():
            dec_raw = pipe.vae.decode(flat, return_dict=False)[0]
        raw_norm = dec_raw / 2 + 0.5
        over = float((raw_norm > 1.0).float().mean())
        under = float((raw_norm < 0.0).float().mean())
        print(f"    decode   raw>1.0: {over*100:.2f}%   raw<0.0: {under*100:.2f}%   "
              f"raw absmax={float(dec_raw.abs().max()):.2f}")
        print(f"    frames   white%>245: {100*(arr>245).mean():.2f}%   "
              f"mean={arr.mean():.1f}  std={arr.std():.1f}")

        rows.append(dict(chunk=ci, skip=skip, steps=len(ts_run),
                         init_absmax=None if is_first else None,
                         final_absmax=fin["absmax"], final_p99=fin["p99"],
                         final_std=fin["std"], final_mean=fin["mean"],
                         vae_in_absmax=tstats(lat_in, "x")["absmax"],
                         white_pct=float(100 * (arr > 245).mean()),
                         raw_over=over))

        keep = max(1, a.overlap)
        prev_tail = lat[:, :, -keep:].clone()

    print(f"\n\n{'='*84}")
    print("CHUNK SUMMARY")
    print(f"{'chunk':>6} {'skip':>5} {'steps':>6} {'lat mean':>10} {'lat std':>9} "
          f"{'lat absmax':>11} {'vae_in absmax':>14} {'raw>1.0%':>9} {'white%':>8}")
    print("-" * 84)
    for r in rows:
        print(f"{r['chunk']:>6} {r['skip']:>5} {r['steps']:>6} {r['final_mean']:>10.4f} "
              f"{r['final_std']:>9.4f} {r['final_absmax']:>11.3f} {r['vae_in_absmax']:>14.3f} "
              f"{r['raw_over']*100:>9.2f} {r['white_pct']:>8.2f}")

    # growth ratios
    if len(rows) >= 2:
        print("\ngrowth vs chunk 0:")
        b0 = rows[0]
        for r in rows[1:]:
            print(f"  chunk {r['chunk']}: std x{r['final_std']/b0['final_std']:.3f}  "
                  f"absmax x{r['final_absmax']/b0['final_absmax']:.3f}  "
                  f"white% {b0['white_pct']:.2f} -> {r['white_pct']:.2f}")


if __name__ == "__main__":
    main()
