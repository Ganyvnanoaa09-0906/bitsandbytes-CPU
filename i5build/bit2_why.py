"""Why does 2-bit destroy the image? Measure quantization error the honest way.

The end-to-end result is unambiguous: SD1.5 renders an astronaut at fp32 and at 4-bit,
and renders noise at 2-bit with per-64 absmax uniform levels. That says the scheme
fails, not why, and it cannot say which of the known fixes would help.

This measures layer by layer, on REAL activations captured from a real forward pass,
using SQNR -- signal energy over quantization-noise energy:

    SQNR = 10 log10( ||WX||^2 / ||(Wq - W)X||^2 )

SQNR is used instead of a relative weight error because a relative error is a ratio of
two scalars and degenerates whenever the reference is near zero or spans orders of
magnitude; an energy ratio over a whole activation matrix does not. This is the same
lesson the pixel comparison taught: for a generative model the per-element error is not
the quantity that decides whether the output survives.

Candidate schemes, all at the shipped 2-bit budget (2 bits + fp32 per-64 absmax =
0.3125 B/weight):
    u2_per64        uniform 4 levels, per-64 absmax          <- what shipped and failed
    u2_perchan      per-output-channel absmax
    u2_per64_had    Hadamard rotation of inputs, then per-64 absmax
    nf2_per64       non-uniform 4-level codebook, per-64 absmax
    u2_per64_had_nf Hadamard rotation + codebook
and, for context, the same measurement at 4 bits.
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn

torch.set_grad_enabled(False)

MODEL = r"D:\work\textmodel\counterfeit-v3-diffusers"
PROMPT = "a photo of an astronaut riding a horse on the moon"


# --- quantizers -------------------------------------------------------------------
# Each returns (dequantized_weight, bytes_per_weight) so no scheme is compared at a
# budget it did not pay for.

def _pack_indices(w, levels, block, per_channel):
    """Scale -> index -> dequantized values, uniform levels."""
    if per_channel:
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        norm = w / scale
        idx = torch.clamp(torch.round((norm + 1) * 0.5 * (levels - 1)), 0, levels - 1)
        dq = (idx / (levels - 1)) * 2 - 1
        return dq * scale, 4.0 / w.shape[1]
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.reshape(-1, block)
    scale = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    norm = blk / scale
    idx = torch.clamp(torch.round((norm + 1) * 0.5 * (levels - 1)), 0, levels - 1)
    dq = ((idx / (levels - 1)) * 2 - 1) * scale
    return dq.reshape(-1)[:n].reshape(w.shape), 4.0 / block


def _pack_codebook(w, centroids, block):
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.reshape(-1, block)
    scale = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    norm = (blk / scale).reshape(-1)
    # nearest centroid
    d = (norm.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    idx = d.argmin(dim=1)
    dq = centroids[idx].reshape(blk.shape) * scale
    return dq.reshape(-1)[:n].reshape(w.shape), 4.0 / block


def fit_codebook(w, block, levels, iters=25, sample=400_000):
    """Lloyd-Max on per-block-normalized weights: the 'nf' in NF2."""
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.reshape(-1, block)
    scale = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    x = (blk / scale).reshape(-1)
    if x.numel() > sample:
        step = x.numel() // sample
        x = x[::step]
    qs = torch.linspace(0, 100, levels + 2)[1:-1]
    c = torch.quantile(x, qs / 100).sort().values
    for _ in range(iters):
        d = (x.unsqueeze(1) - c.unsqueeze(0)).abs()
        idx = d.argmin(dim=1)
        for k in range(levels):
            m = idx == k
            if m.any():
                c[k] = x[m].mean()
        c = c.sort().values
    return c


def hadamard(n):
    """Block-diagonal normalized Hadamard on the largest power of two dividing n."""
    b = 1
    while n % (b * 2) == 0 and b < 512:
        b *= 2
    if b < 2:
        return None, 1
    h = torch.ones(1, 1)
    while h.shape[0] < b:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = h / math.sqrt(b)
    H = torch.zeros(n, n)
    for i in range(0, n, b):
        H[i:i + b, i:i + b] = h
    return H, b


# --- capture ----------------------------------------------------------------------

def capture_activations(unet, sample, t, ctx):
    acts = {}

    def mk(name):
        def hook(mod, inp, out):
            acts[name] = inp[0].detach().to(torch.float32).reshape(-1, inp[0].shape[-1])
        return hook

    hs = []
    for name, mod in unet.named_modules():
        if isinstance(mod, nn.Linear):
            hs.append(mod.register_forward_hook(mk(name)))
    t0 = time.time()
    unet(sample, t, encoder_hidden_states=ctx)
    dt = time.time() - t0
    for h in hs:
        h.remove()
    return acts, dt


def sqnr(y, yq):
    num = (y ** 2).sum().item()
    den = ((yq - y) ** 2).sum().item()
    if den <= 0:
        return float("inf")
    return 10.0 * math.log10(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--layers", type=int, default=0, help="0 = all")
    ap.add_argument("--bits", type=int, default=2)
    a = ap.parse_args()

    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"model  : {a.model}")
    print(f"prompt : {a.prompt!r}")
    print(f"res    : {a.res}x{a.res}   budget bits={a.bits}")
    print(flush=True)

    tok = CLIPTokenizer.from_pretrained(os.path.join(a.model, "tokenizer"))
    te = CLIPTextModel.from_pretrained(os.path.join(a.model, "text_encoder"),
                                       torch_dtype=torch.float32).eval()
    unet = UNet2DConditionModel.from_pretrained(os.path.join(a.model, "unet"),
                                                torch_dtype=torch.float32).eval()

    ids = tok(a.prompt, return_tensors="pt", padding="max_length",
              truncation=True, max_length=77).input_ids
    ctx = te(ids)[0]
    lat = a.res // 8
    sample = torch.randn(1, 4, lat, lat) * 1.0

    print("capturing real activations (one forward pass) ...", flush=True)
    acts, dt = capture_activations(unet, sample, torch.tensor([500.0]), ctx)
    print(f"  forward pass {dt:.1f} s, captured {len(acts)} Linear layers", flush=True)

    sd = unet.state_dict()
    rows = []
    t0 = time.time()
    for name, X in acts.items():
        W = sd.get(name + ".weight")
        if W is None:
            continue
        W = W.to(torch.float32)
        if a.layers and len(rows) >= a.layers:
            break
        if min(W.shape) < 8 or X.shape[0] < 4:
            continue
        out_f, in_f = W.shape
        Y = X @ W.t()
        H, blk = hadamard(in_f)
        Xr = X @ H if H is not None else X
        Wr = W @ H if H is not None else W

        res = {}
        levels = 2 ** a.bits
        Wq, bpw = _pack_indices(W, levels, 64, False)
        res["u%d_per64" % a.bits] = (sqnr(Y, X @ Wq.t()), bpw)
        Wq, bpw = _pack_indices(W, levels, 64, True)
        res["u%d_perchan" % a.bits] = (sqnr(Y, X @ Wq.t()), bpw)
        if H is not None:
            Wq, bpw = _pack_indices(Wr, levels, 64, False)
            # W' = W H is stored, the input arrives already rotated as X' = X H, and
            # W' X' = W H H^T X = W X. The rotation is inside Wq, so nothing is applied
            # to the output: an extra H here is the dimension mismatch this line used
            # to raise.
            res["u%d_per64_had" % a.bits] = (sqnr(Y, Xr @ Wq.t()), bpw)
        cb = fit_codebook(W, 64, levels)
        Wq, bpw = _pack_codebook(W, cb, 64)
        res["nf%d_per64" % a.bits] = (sqnr(Y, X @ Wq.t()), bpw)
        if H is not None:
            cbr = fit_codebook(Wr, 64, levels)
            Wq, bpw = _pack_codebook(Wr, cbr, 64)
            res["u%d_had_nf" % a.bits] = (sqnr(Y, Xr @ Wq.t()), bpw)
        # 4-bit reference
        Wq, bpw4 = _pack_indices(W, 16, 64, False)
        res["u4_per64_ref"] = (sqnr(Y, X @ Wq.t()), bpw4)

        # distribution shape of the raw weights
        wf = W.reshape(-1)
        mu, sd_ = wf.mean(), wf.std()
        kurt = (((wf - mu) / sd_) ** 4).mean().item()
        amax_ratio = (wf.abs().max() / wf.abs().median()).item()
        rows.append(dict(name=name, params=W.numel(), kurt=kurt,
                         amax_ratio=amax_ratio, **{k: v[0] for k, v in res.items()},
                         bpw=res["u%d_per64" % a.bits][1]))
        del W, Wr, Y, Xr
    print(f"  measured {len(rows)} layers in {time.time()-t0:.1f} s", flush=True)

    keys = ["u%d_per64" % a.bits, "u%d_perchan" % a.bits, "u%d_per64_had" % a.bits,
            "nf%d_per64" % a.bits, "u%d_had_nf" % a.bits, "u4_per64_ref"]
    keys = [k for k in keys if any(k in r for r in rows)]
    # Bytes per weight actually paid: the levels plus the fp32 scale, which is 4/64 B
    # per weight for every per-block scheme here. The first version of this table
    # printed only the scale overhead for 2-bit and a hard-coded 1.125 for 4-bit.
    bpw_of = {k: (a.bits / 8 + 4 / 64) if k.startswith(("u%d" % a.bits, "nf"))
              else (4 / 8 + 4 / 64) for k in keys}

    tot = sum(r["params"] for r in rows)
    print(f"\n=== SQNR (dB), parameter-weighted over {len(rows)} layers, "
          f"{tot/1e6:.1f} M params ===")
    print(f"{'scheme':<16} {'B/weight':>9} {'weighted SQNR':>14} {'worst layer':>12}")
    base = None
    for k in keys:
        vals = [(r[k], r["params"]) for r in rows if k in r]
        wsum = sum(v * p for v, p in vals) / sum(p for _, p in vals)
        worst = min(v for v, _ in vals)
        print(f"{k:<16} {bpw_of[k]:9.4f} {wsum:14.2f} {worst:12.2f}")
        if k == "u%d_per64" % a.bits:
            base = wsum

    print("\n=== gain over u%d_per64 (what shipped and failed) ===" % a.bits)
    for k in keys[1:]:
        vals = [(r[k], r["params"]) for r in rows if k in r]
        wsum = sum(v * p for v, p in vals) / sum(p for _, p in vals)
        delta = wsum - base
        print(f"  {k:<16} {delta:+7.2f} dB   ({'better' if delta > 0 else 'worse'})")

    print("\n=== weight distribution shape (are there outliers to rotate away?) ===")
    ks = [r["kurt"] for r in rows]
    ar = [r["amax_ratio"] for r in rows]
    print(f"  kurtosis   : median {sorted(ks)[len(ks)//2]:8.2f}   "
          f"(Gaussian = 3.0)   min {min(ks):.2f}  max {max(ks):.2f}")
    print(f"  max/median : median {sorted(ar)[len(ar)//2]:8.1f}   "
          f"min {min(ar):.1f}  max {max(ar):.1f}")

    print("\n=== worst 12 layers at u%d_per64 ===" % a.bits)
    k0 = "u%d_per64" % a.bits
    for r in sorted(rows, key=lambda r: r[k0])[:12]:
        extra = " ".join(f"{k.split('_')[0]}={r[k]:6.2f}" for k in keys[1:] if k in r)
        print(f"  {r[k0]:7.2f} dB  {r['params']/1e6:5.2f}M  kurt={r['kurt']:6.1f}  "
              f"{r['name'][:44]:<44} {extra}")

    import json
    with open("bit2_why_results.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\nwrote bit2_why_results.json")


if __name__ == "__main__":
    main()
