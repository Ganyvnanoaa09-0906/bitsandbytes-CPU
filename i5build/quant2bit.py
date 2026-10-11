"""quant2bit.py -- a 2-bit blockwise quantizer in pure torch, with no kernel needed.

Why this exists at all
----------------------
Everything measured so far says 2-bit does not buy speed on this machine: at M=1 the
shipped 4-bit kernel is already at 24.75 GB/s against a 21.56 GB/s memcpy ceiling, so
there is no bandwidth left to save, and at M>1 the work is compute bound so bit width
does not matter (1.01-1.05x across a format-only comparison).

What 2-bit would still buy is SIZE. The text encoder is 8.76e9 parameters and 4.9 GB
at 4 bits; halving that is the difference between fitting and not fitting on a small
machine. So this file answers the question that actually decides it: what does the
precision cost, measured properly.

The scheme
----------
Four levels, uniform, at {-1, -1/3, +1/3, +1} times a per-block absmax, blocksize 64 to
match the 4-bit format. Uniform levels are what make the decode cheap -- level[c] =
(c-1)/3 -- and they are also the thing most likely to cost precision, because NF4's
sixteen levels were chosen to be optimal for normally distributed weights and four
evenly spaced levels were not.

Quantization is pure torch, so there is no kernel to be wrong about:

    idx = clamp(round(w / absmax * 1.5 + 1.5), 0, 3)

Packing: four 2-bit indices per byte, low field first, matching the convention the
kernels in this project already use: code j of byte b is (b >> 2j) & 3.

What is measured
----------------
  * reconstruction error on synthetic Gaussian weights, against the 4-bit NF4 error
    measured the same way, so the two are comparable
  * the same on REAL tensors taken from the checkpoint, since synthetic Gaussian data
    flatters or penalises a quantizer depending on the actual weight distribution
  * the size, exactly rather than by projection

Usage:
    python quant2bit.py [--model <bf16 dir>] [--tensors n]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

BLOCKSIZE = 64
# level[c] = (c - 1) / 3 for c in 0..3
LEVELS = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=torch.float32)


def quant2bit(w, blocksize=BLOCKSIZE):
    """Quantize a 2-D float tensor to 2 bits per weight. Returns (packed, absmax).

    Rows and the flattened tensor are both handled: the block loop runs over the
    flattened tensor, which is what a blockwise quantizer means, and the shape is
    kept alongside so the caller can reshape.
    """
    flat = w.detach().float().reshape(-1)
    n = flat.numel()
    pad = (-n) % blocksize
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.reshape(-1, blocksize)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)          # per block
    norm = blocks / absmax.unsqueeze(1)                          # in [-1, 1]
    # map [-1,1] onto 0..3: idx = round(norm*1.5 + 1.5)
    idx = torch.clamp(torch.round(norm * 1.5 + 1.5), 0, 3).to(torch.uint8)
    idx = idx.reshape(-1)[:n]
    # pack four indices per byte, low field first
    ppad = (-idx.numel()) % 4
    if ppad:
        idx = torch.cat([idx, torch.zeros(ppad, dtype=torch.uint8)])
    g = idx.reshape(-1, 4).to(torch.int32)
    packed = (g[:, 0] | (g[:, 1] << 2) | (g[:, 2] << 4) | (g[:, 3] << 6)).to(torch.uint8)
    return packed, absmax, tuple(w.shape)


def dequant2bit(packed, absmax, shape, blocksize=BLOCKSIZE):
    """Inverse of quant2bit, for measuring error. Not a kernel -- a reference."""
    p = packed.to(torch.int32)
    idx = torch.empty(p.numel() * 4, dtype=torch.uint8)
    idx[0::4] = (p & 3).to(torch.uint8)
    idx[1::4] = ((p >> 2) & 3).to(torch.uint8)
    idx[2::4] = ((p >> 4) & 3).to(torch.uint8)
    idx[3::4] = ((p >> 6) & 3).to(torch.uint8)
    n = 1
    for d in shape:
        n *= d
    idx = idx[:n].to(torch.int64)
    lv = LEVELS.to(torch.float32)[idx]
    blocks = lv.reshape(-1, blocksize) * absmax.unsqueeze(1)
    return blocks.reshape(-1)[:n].reshape(shape)


def quant4bit_nf4(w, blocksize=BLOCKSIZE, chunk=1 << 22):
    """The 4-bit NF4 reference, using bitsandbytes' own codebook.

    Memory discipline, which two earlier versions got wrong:
      * the nearest-codebook search loops over the sixteen code values instead of
        broadcasting (norm, 16), which on the text encoder's embedding asks for 39.8 GB
      * the two accumulators are updated IN PLACE and the replacement value is a
        Python float, not torch.full_like(norm, c) -- the latter allocates a fresh
        full-size tensor on every one of the sixteen iterations
      * the work is done in chunks, because even with the above the live set is
        several full-size float32 vectors and the embedding alone is 2.49 GB

    The result is unchanged by any of this; the reference exists to be trusted, not
    to be quick.
    """
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))
    import bitsandbytes as bnb

    flat = w.detach().float().reshape(-1)
    n = flat.numel()
    pad = (-n) % blocksize
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.reshape(-1, blocksize)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    code = bnb.functional.get_4bit_type("nf4", device="cpu").to(torch.float32).reshape(-1).tolist()

    out_blocks = torch.empty_like(blocks)
    nb = blocks.shape[0]
    for s in range(0, nb, chunk):
        e = min(nb, s + chunk)
        blk = blocks[s:e]
        am = absmax[s:e].unsqueeze(1)
        norm = blk / am
        best = norm.abs().clone()          # distance to the first code (not 0)
        best.fill_(float("inf"))
        lv = torch.zeros_like(norm)
        for c in code:
            d = norm - c
            d.abs_()                       # in place, no second allocation
            take = d < best
            best.copy_(torch.where(take, d, best))
            lv.copy_(torch.where(take, float(c), lv))
            del d, take
        out_blocks[s:e] = lv * am
        del blk, am, norm, best, lv
    out = out_blocks.reshape(-1)[:n]
    return out.reshape(w.shape)


def quant8bit(w, blocksize=BLOCKSIZE):
    """Int8 blockwise, 256 levels with a per-block absmax. The other end of the range.

    Included because "mix 8 and 4" is a request to reduce error rather than size, and
    the only way to test it is to have all three widths measured on the same tensors
    with the same block scheme.
    """
    flat = w.detach().float().reshape(-1)
    n = flat.numel()
    pad = (-n) % blocksize
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.reshape(-1, blocksize)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    q = torch.clamp(torch.round(blocks / absmax.unsqueeze(1) * 127.0), -127, 127)
    rec = (q / 127.0 * absmax.unsqueeze(1)).reshape(-1)[:n]
    return rec.reshape(w.shape)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--tensors", type=int, default=8)
    a = ap.parse_args()

    print(f"blocksize {BLOCKSIZE}, levels {LEVELS.tolist()}")
    print()

    # ---------- synthetic ----------
    print("=" * 74)
    print("synthetic Gaussian weights (this is what the 4-bit codebook was designed for,")
    print("so it is the case most likely to favour NF4)")
    print("=" * 74)
    torch.manual_seed(0)
    for shape, label in (((4096, 4096), "square 4096^2"),
                         ((12288, 4096), "MLP 12288x4096"),
                         ((128, 4096), "narrow 128x4096")):
        w = torch.randn(*shape, dtype=torch.float32)
        q2, am2, shp = quant2bit(w)
        d2 = dequant2bit(q2, am2, shp)
        d4 = quant4bit_nf4(w)
        b16 = w.to(torch.bfloat16).float()
        n = w.numel()
        print(f"  {label}")
        print(f"    2-bit : rel err {rel(d2, w):.6f}   "
              f"bytes/weight {q2.numel()/n + am2.numel()*4/n:.4f}")
        print(f"    4-bit : rel err {rel(d4, w):.6f}   bytes/weight ~0.56")
        print(f"    bf16  : rel err {rel(b16, w):.6f}   bytes/weight 2.0000")
    print()

    # ---------- real tensors ----------
    print("=" * 74)
    print("real tensors from the checkpoint (the weight distribution is what matters, and")
    print("synthetic Gaussian data does not have it)")
    print("=" * 74)
    from safetensors import safe_open

    def shards_for(component, prefix):
        idx_path = os.path.join(a.model, component,
                                f"{prefix}.safetensors.index.json")
        out = {}
        if os.path.isfile(idx_path):
            with open(idx_path, encoding="utf-8") as fh:
                out = json.load(fh)["weight_map"]
            out = {k: v for k, v in out.items() if k.endswith(".weight")}
        return out, idx_path

    # Prefer the transformer, but fall back to whatever is actually on disk: the model
    # is being re-downloaded and a missing shard is a fact to report, not a crash.
    sources = [("transformer", "diffusion_pytorch_model"),
               ("text_encoder", "model")]
    wm = {}
    used = None
    for component, prefix in sources:
        cand, idx_path = shards_for(component, prefix)
        if not cand:
            print(f"  ({component}: no index at {idx_path})")
            continue
        present = {k: v for k, v in cand.items()
                   if os.path.isfile(os.path.join(a.model, component, v))}
        print(f"  {component}: {len(cand)} weight names in the index, "
              f"{len(present)} on a shard that is present")
        if present:
            wm = {k: (component, v) for k, v in present.items()}
            used = component
            break
    if not wm:
        print("  no component has a readable shard yet; skipping")
        return 0

    cands = sorted(wm)
    step = max(1, len(cands) // a.tensors)
    picks = cands[::step][:a.tensors]

    print(f"  sampling {len(picks)} tensors from {used}")
    print()
    print(f"  {'tensor':52s} {'shape':>16s} {'2-bit':>9s} {'4-bit':>9s} {'ratio':>7s}")
    tot2 = tot4 = 0.0
    counted = 0
    cur = None
    fh = None
    for k in picks:
        comp, shard = wm[k]
        if shard != cur:
            if fh is not None:
                fh.__exit__(None, None, None)
                fh = None
            cur = shard
            fh = safe_open(os.path.join(a.model, comp, shard),
                           framework="pt", device="cpu")
            fh = fh.__enter__()
        try:
            w = fh.get_tensor(k)
        except Exception as e:  # noqa: BLE001
            print(f"  {k[:52]:52s} could not read: {type(e).__name__}")
            continue
        if w.dim() != 2:
            continue
        for_dequant = w
        packed, absmax, shape = quant2bit(w)
        e2 = rel(dequant2bit(packed, absmax, shape), for_dequant)
        e4 = rel(quant4bit_nf4(w), for_dequant)
        tot2 += e2
        tot4 += e4
        counted += 1
        print(f"  {k[:52]:52s} {str(tuple(w.shape)):>16s} {e2:9.6f} {e4:9.6f} "
              f"{e2/max(1e-12,e4):6.2f}x")
    if fh is not None:
        fh.__exit__(None, None, None)
    print()
    if counted:
        print(f"  mean over {counted} sampled tensors: 2-bit {tot2/counted:.6f}  "
              f"4-bit {tot4/counted:.6f}  "
              f"ratio {(tot2/counted)/max(1e-12,(tot4/counted)):.2f}x")
    print()
    print("reading:")
    print("  the ratio column is what 2-bit costs relative to NF4 on the same tensor.")
    print("  A ratio near 1 means the four uniform levels happen to suit these weights;")
    print("  a ratio of 5 means 2-bit error is five times NF4's, which is the figure")
    print("  the report quotes for a different scheme and which this measures directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
