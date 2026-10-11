# -*- coding: utf-8 -*-
"""Independent-chunk generation: every chunk is a FULL generation, joined by hard cuts.

Motivation (measured): the sliding-window relay starts each carried chunk at t=400 and runs
only 2 steps, so it can only *refine* content, never *create* detail. Result: chunk detail
plateaus at ~72-103 (Laplacian) versus ~164 for a full 5-step chunk -- a ~50% loss that
adding relay steps does not fix (4 steps buys only 9 points while tripling the latent drift).

For 漫剧 (anime drama) the alternative is to stop pretending this is a single continuous
take: generate each shot as an independent full-schedule sample and join them with hard
cuts, which is normal editing language for the medium.

Each chunk here:
  * starts from pure noise (t=999),
  * runs the COMPLETE schedule,
  * uses its own seed,
so every chunk should come out at full detail with no cross-chunk decay.

Reports per-chunk Laplacian variance (expected to stay flat, unlike the relay's decay) and
the seam MAD (expected to be HIGH -- hard cuts are intentional, not a defect).

Usage:
  python independent_chunks.py --res 512 --chunk 10 --chunks 3 --steps 5 --tag shots512
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from anime_adiff import build, ANIME_PROMPT, ANIME_NEG, OUT_DIR, VAE_DIR  # noqa: E402

GUIDANCE = 7.5


def lapvar(a):
    a = a.astype(np.float32)
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def fstats(a):
    a = a.astype(np.float32)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    return dict(lap=lapvar(a), white=float(100 * (a > 245).mean()),
                blown=float(((lum > 235) & (sat < 0.08)).mean() * 100),
                green=int(((g - np.maximum(r, b)) > 25).sum()),
                mean=float(a.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt", default=ANIME_PROMPT)
    ap.add_argument("--neg", default=ANIME_NEG)
    ap.add_argument("--tag", default="shots")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--lora", nargs="*", default=[],
                    help="zero or more LoRA dirs/files, merged into the UNet in order")
    ap.add_argument("--lora-scale", nargs="*", type=float, default=[],
                    help="scale per LoRA, positionally matched (default 0.8 each)")
    ap.add_argument("--lora-method", default="manual", choices=["auto", "peft", "manual"],
                    help="manual = merge W += scale*B@A ourselves (no peft needed). "
                         "'auto' tries peft first -- NOTE that a FAILED peft attempt may "
                         "leave the pipeline half-modified, which is the suspected cause of "
                         "the earlier garbage output, so manual is the default.")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    outdir = a.outdir or os.path.join(OUT_DIR, f"{a.tag}_{a.res}_{a.chunk}f{a.chunks}n")
    os.makedirs(outdir, exist_ok=True)

    print(f"[cfg] res={a.res} chunk={a.chunk} chunks={a.chunks} steps={a.steps} "
          f"mode=INDEPENDENT (full schedule per chunk, hard cuts)")
    print(f"[cfg] vae={VAE_DIR}")

    pipe = build("dpm")

    def merge_lora(path, lora_scale):
        """W += lora_scale * (alpha/r) * (B @ A) for every adapter pair found in `path`."""
        ok = False
        if a.lora_method in ("auto", "peft"):
            try:
                pipe.load_lora_weights(path, adapter_name=f"a{len(_done)}")
                pipe.set_adapters([f"a{len(_done)}"], adapter_weights=[lora_scale])
                print(f"[cfg] lora loaded via peft: {path} (scale={lora_scale})")
                ok = True
            except Exception as e:
                print(f"[info] peft path unavailable ({str(e)[:80]})")
                if a.lora_method == "peft":
                    raise
        if not ok:
            # .venv_dml has no peft, so diffusers' load_lora_weights cannot be used. LoRA is
            # just W += scale * (B @ A), so merge it straight into the UNet weights instead
            # of installing peft (which would risk the torch 2.4.1 / torch_directml pin).
            import json as _json
            from safetensors.torch import load_file as _lf
            p = path
            if os.path.isdir(p):
                sd = _lf(os.path.join(p, "adapter_model.safetensors"))
                cfgp = os.path.join(p, "adapter_config.json")
                alpha = r = None
                if os.path.exists(cfgp):
                    c = _json.load(open(cfgp, encoding="utf-8"))
                    alpha, r = c.get("lora_alpha"), c.get("r")
            else:
                sd = _lf(p)
                alpha = r = None
            scale = lora_scale * ((alpha / r) if (alpha and r) else 1.0)
            mods = dict(pipe.unet.named_modules())
            pairs, applied = {}, 0
            for k, v in sd.items():
                if not k.startswith("base_model.model."):
                    continue
                rest = k[len("base_model.model."):]
                if rest.endswith(".lora_A.weight"):
                    pairs.setdefault(rest[:-len(".lora_A.weight")], {})["A"] = v
                elif rest.endswith(".lora_B.weight"):
                    pairs.setdefault(rest[:-len(".lora_B.weight")], {})["B"] = v
            for path_, ab in pairs.items():
                if "A" not in ab or "B" not in ab or path_ not in mods:
                    continue
                mod = mods[path_]
                if not hasattr(mod, "weight"):
                    continue
                delta = (ab["B"].float() @ ab["A"].float()) * scale
                with torch.no_grad():
                    mod.weight.add_(delta.to(mod.weight.dtype))
                applied += 1
            print(f"[cfg] lora merged manually: {applied} layers, scale={scale:.3f} "
                  f"(alpha={alpha}, r={r}) <- {path}")
            return applied
        return -1

    _done = []
    if a.lora:
        scales = list(a.lora_scale)
        while len(scales) < len(a.lora):
            scales.append(scales[-1] if scales else 0.8)
        for lp, ls in zip(a.lora, scales):
            merge_lora(lp, ls)
            _done.append(lp)
        print(f"[cfg] {len(_done)} LoRA(s) merged; total delta is the SUM of all adapters")
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, "cpu", 1, True, negative_prompt=a.neg)
    embeds = torch.cat([ne, pe])

    lat_h = a.res // pipe.vae_scale_factor
    sched = pipe.scheduler
    per_chunk = []
    all_frames = []
    t_all = time.time()

    for ci in range(a.chunks):
        sched.set_timesteps(a.steps)
        sched._step_index = 0
        sched.model_outputs = [None] * sched.config.solver_order
        sched._lower_order_nums = 0
        ts_run = list(sched.timesteps)

        gen = torch.Generator().manual_seed(a.seed + ci * 1000)
        lat = torch.randn((1, pipe.unet.config.in_channels, a.chunk, lat_h, lat_h),
                          generator=gen, dtype=torch.float32)
        lat = lat * sched.init_noise_sigma
        t0 = time.time()
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
            flat = (lat / pipe.vae.config.scaling_factor).permute(0, 2, 1, 3, 4
                                                                  ).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat, return_dict=False)[0]
            arr = (((dec / 2 + 0.5).clamp(0, 1)).reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                   .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype(np.uint8)

        st = [fstats(arr[i]) for i in range(len(arr))]
        rec = dict(chunk=ci, seed=a.seed + ci * 1000, steps=len(ts_run),
                   seconds=time.time() - t0,
                   lap=float(np.mean([s["lap"] for s in st])),
                   white=float(np.mean([s["white"] for s in st])),
                   blown=float(np.mean([s["blown"] for s in st])),
                   green=int(np.sum([s["green"] for s in st])),
                   mean=float(np.mean([s["mean"] for s in st])))
        per_chunk.append(rec)
        print(f"  [shot {ci}] {rec['seconds']:.0f}s  lap={rec['lap']:>7.1f}  "
              f"white%={rec['white']:.2f}  green={rec['green']:>4}  mean={rec['mean']:.1f}",
              flush=True)
        all_frames.extend(list(arr))

    dt = time.time() - t_all

    fdir = os.path.join(outdir, "frames")
    os.makedirs(fdir, exist_ok=True)
    imgs = [Image.fromarray(f).convert("RGB") for f in all_frames]
    for i, im in enumerate(imgs):
        im.save(os.path.join(fdir, f"f{i:03d}.png"))
    gif = os.path.join(outdir, f"{a.tag}_{a.res}_{a.chunk}f{a.chunks}n_{a.steps}s.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=200, loop=0)

    laps = [r["lap"] for r in per_chunk]
    fr = [f.astype(np.float32) for f in all_frames]
    mads = [float(np.abs(fr[i + 1] - fr[i]).mean()) for i in range(len(fr) - 1)]
    seams = [mads[i] for i in range(len(mads)) if (i + 1) % a.chunk == 0]
    intra = [mads[i] for i in range(len(mads)) if (i + 1) % a.chunk != 0]

    print(f"\n{'='*72}")
    print(f"RESULT  independent shots  ({len(all_frames)} frames, {dt:.0f}s total)")
    print(f"  Laplacian per shot : " + " ".join(f"{v:.1f}" for v in laps))
    print(f"  detail spread      : {min(laps):.1f} .. {max(laps):.1f}  "
          f"(relay decayed 164 -> 72; independent should stay flat)")
    print(f"  white% per shot    : " + " ".join(f"{r['white']:.2f}" for r in per_chunk))
    print(f"  green px total     : {sum(r['green'] for r in per_chunk)}")
    print(f"  intra-shot MAD     : {np.mean(intra):.2f}")
    print(f"  SEAM MAD (cuts)    : {np.mean(seams) if seams else float('nan'):.2f}"
          f"   <- intentionally high = hard cut")
    print(f"  seconds per frame  : {dt/len(all_frames):.1f}")
    print(f"[save] {gif}")
    print(f"[save] {fdir}\\f000..f{len(imgs)-1:03d}.png")

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(config=vars(a), vae=VAE_DIR, per_chunk=per_chunk,
                       lap=laps, intra_mad=float(np.mean(intra)),
                       seam_mad=float(np.mean(seams)) if seams else None,
                       seconds=dt), fh, indent=2)


if __name__ == "__main__":
    main()
