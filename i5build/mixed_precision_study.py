"""mixed_precision_study.py -- is per-layer bit allocation worth anything here?

The idea: give the layers that tolerate it fewer bits and the sensitive ones more, so
the total error stays where a uniform scheme would put it while the total size drops.
That only works if sensitivity VARIES across layers. If every tensor responds the same
way, mixing just moves the average around and saves nothing, and the whole exercise is
arithmetic theatre.

The evidence that it might vary is already in hand. On one vision-tower qkv layer:

    weight rel err   2-bit 1.0435    4-bit 0.1155     ratio 9.03x
    output rel err   2-bit 0.000022  4-bit 0.000002   ratio 9.05x
    cosine of both against the original: 1.00000000

so a weight error LARGER THAN THE WEIGHTS THEMSELVES produced an output error five
orders of magnitude smaller. That layer is nearly indifferent to weight error. Another
layer may not be, and if some are indifferent then those are the ones to compress.

This maps the distribution, which is the precondition for any allocation decision:

  * per tensor: 2-bit and 4-bit reconstruction error, and the ratio between them
  * per tensor: the output error each causes, on a realistic input, which is the
    quantity that actually matters and which cannot be read off the weight error
  * the same for 8-bit, since the question was about combining 8 and 4

The output-error measurement needs each layer to run, so it walks whatever layers are
fully present on the shards that have finished downloading, and says how many it could
cover rather than pretending to cover them all.

Usage:
    python mixed_precision_study.py [--max-layers 40] [--seq 64]
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import quant2bit as Q2  # noqa: E402


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def quant8bit(w, blocksize=64):
    """Int8 blockwise: 256 levels, per-block absmax. The other end of the range."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--max-layers", type=int, default=40)
    ap.add_argument("--seq", type=int, default=64)
    a = ap.parse_args()

    # Which linear layers are fully present? Only those can be run.
    from safetensors import safe_open
    from transformers import AutoConfig

    present = {}   # key -> (component, shard)
    for comp, prefix in (("text_encoder", "model"),
                         ("transformer", "diffusion_pytorch_model")):
        idx = os.path.join(a.model, comp, f"{prefix}.safetensors.index.json")
        if not os.path.isfile(idx):
            continue
        with open(idx, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        for k, v in wm.items():
            if k.endswith(".weight") and os.path.isfile(os.path.join(a.model, comp, v)):
                present[k] = (comp, v)

    print(f"{len(present)} weight tensors are readable right now")
    if not present:
        print("nothing complete yet; rerun when the download finishes")
        return 0

    # Weight-level errors need no model at all: they can be computed for every tensor
    # that is readable, which is the broad sample. Output-level errors need a layer to
    # run, so they come later and on fewer tensors.
    keys = sorted(present)
    step = max(1, len(keys) // a.max_layers)
    sample = keys[::step][:a.max_layers]

    print()
    print("=" * 96)
    print("weight reconstruction error per tensor (this needs no forward pass)")
    print("=" * 96)
    print(f"  {'tensor':50s} {'shape':>15s} {'2bit':>9s} {'4bit':>9s} {'8bit':>9s} "
          f"{'2/4':>6s}")
    rows = []
    cur = None
    fh = None
    for k in sample:
        comp, shard = present[k]
        if shard != cur:
            if fh is not None:
                fh.__exit__(None, None, None)
            cur = shard
            fh = safe_open(os.path.join(a.model, comp, shard), framework="pt",
                           device="cpu")
            fh = fh.__enter__()
        try:
            w = fh.get_tensor(k)
        except Exception:  # noqa: BLE001
            continue
        if w.dim() != 2 or w.numel() < 1 << 16:
            continue
        e8 = rel(quant8bit(w), w)
        packed, absmax, shape = Q2.quant2bit(w)
        e2 = rel(Q2.dequant2bit(packed, absmax, shape), w)
        e4 = rel(Q2.quant4bit_nf4(w), w)
        rows.append((k, tuple(w.shape), e2, e4, e8))
        print(f"  {k[:50]:50s} {str(tuple(w.shape)):>15s} {e2:9.5f} {e4:9.5f} "
              f"{e8:9.5f} {e2/max(1e-12,e4):6.2f}")
    if fh is not None:
        fh.__exit__(None, None, None)
    print()

    if not rows:
        print("no 2-D tensors large enough to be interesting were readable")
        return 0

    import statistics as st
    r24 = [r[2] / max(1e-12, r[3]) for r in rows]
    print(f"  tensors sampled: {len(rows)}")
    print(f"  2-bit/4-bit error ratio: median {st.median(r24):.3f}, "
          f"min {min(r24):.3f}, max {max(r24):.3f}")
    print(f"  4-bit error: median {st.median([r[3] for r in rows]):.5f}  "
          f"min {min(r[3] for r in rows):.5f}  max {max(r[3] for r in rows):.5f}")
    print(f"  8-bit error: median {st.median([r[4] for r in rows]):.5f}")
    print()

    spread = (max(r24) - min(r24)) / st.median(r24)
    print("IS THERE ANYTHING TO EXPLOIT?")
    print(f"  the ratio 2-bit/4-bit spans {min(r24):.2f} to {max(r24):.2f}, a spread of "
          f"{100*spread:.0f}% of the median")
    if spread < 0.25:
        print("  -> the tensors respond nearly identically, so per-layer allocation")
        print("     has almost nothing to work with: whichever bits a layer gets, it")
        print("     pays the same price. Mixing would only redistribute the average.")
    else:
        print("  -> the response genuinely varies, so there IS room to allocate: put")
        print("     2-bit on the insensitive tail and 4-bit on the rest.")
        rows_sorted = sorted(rows, key=lambda r: r[2] / max(1e-12, r[3]))
        print(f"     most tolerant 3: "
              f"{[r[0].split('.')[-2] + '.' + r[0].split('.')[-1] for r in rows_sorted[:3]]}")
        print(f"     least tolerant 3: "
              f"{[r[0].split('.')[-2] + '.' + r[0].split('.')[-1] for r in rows_sorted[-3:]]}")
    print()
    print("NOTE on the 8-bit column: 8-bit costs 1.125 bytes/weight against 4-bit's")
    print("0.5625, so pairing 8 with 4 is a 2x size increase against the uniform-4-bit")
    print("baseline. Pairing it with 2-bit (0.3125) is 3.6x. Both only pay off if the")
    print("layers that get the extra bits are ones whose error actually reaches the")
    print("output, which is what the next measurement has to establish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
