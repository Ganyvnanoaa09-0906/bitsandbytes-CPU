"""pareto_bits.py -- the size/error frontier for mixing bit widths, with 8-bit included.

The question being answered
---------------------------
"Mix 8 and 4" is a request to REDUCE error, not to reduce size. So the useful output is
a curve: for each affordable size, what is the least error achievable by assigning each
tensor 2, 4 or 8 bits? Every point on that curve is a real assignment, not a rule of
thumb, and the curve shows whether mixing beats uniform at all.

The three widths, with per-block absmax included:

    2-bit  0.3125 bytes/weight   error ~0.53
    4-bit  0.5625 bytes/weight   error ~0.093
    8-bit  1.1250 bytes/weight   error ~0.006

Two facts already measured that shape the answer:

  * 2 -> 4 costs 0.25 bytes/weight and removes (0.53 - 0.093) = 0.44 of error
  * 4 -> 8 costs 0.5625 bytes/weight and removes (0.093 - 0.006) = 0.087 of error

so 2 -> 4 removes five times as much error per byte as 4 -> 8. Upgrading looks far
worse than downgrading, which is why a request to reduce error with 8-bit has to be
tested rather than assumed to help.

The allocator
-------------
A three-way greedy on the size/error plane. For each tensor the three candidate
assignments are known, so the frontier is built by repeatedly taking the step with the
best error reduction per byte spent, starting from everything at 2-bit. This is the
Lagrangian/greedy solution and it is exact for independent items with linear
contributions, which is what a sum of per-tensor errors is.

The error metric is the weighted mean relative reconstruction error. Weighting matters
and two are reported:
  * by parameter count -- the straightforward one
  * by row-norm sum -- a proxy for how much signal the tensor carries, so that error in
    a tensor feeding large activations counts for more

The honest caveat stays attached: this is weight error, not image quality. One measured
layer turned a weight error of 1.04 into an output error of 2.2e-05, so the two are not
proportional and the conversion has to be measured per layer.

Usage:
    python pareto_bits.py [--max-tensors N] [--weights params|signal|both]
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import quant2bit as Q2  # noqa: E402

BITS = (2, 4, 8)


def bpw(bits, blocksize=64):
    return bits / 8.0 + 4.0 / blocksize


def measure(model_dir, components, max_tensors=0, verbose=True):
    from safetensors import safe_open

    index = {}
    for comp, prefix in components:
        idx = os.path.join(model_dir, comp, f"{prefix}.safetensors.index.json")
        if not os.path.isfile(idx):
            continue
        with open(idx, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        for k, v in wm.items():
            if k.endswith(".weight") and os.path.isfile(os.path.join(model_dir, comp, v)):
                index[k] = (comp, v)
    if verbose:
        print(f"  {len(index)} readable weight tensors")
    rows = []
    cur = None
    fh = None
    t0 = time.time()
    for k in sorted(index):
        comp, shard = index[k]
        if shard != cur:
            if fh is not None:
                fh.__exit__(None, None, None)
            cur = shard
            fh = safe_open(os.path.join(model_dir, comp, shard), framework="pt",
                           device="cpu")
            fh = fh.__enter__()
        try:
            w = fh.get_tensor(k)
        except Exception:  # noqa: BLE001
            continue
        if w.dim() != 2:
            continue
        packed, absmax, shape = Q2.quant2bit(w)
        e2 = Q2.rel(Q2.dequant2bit(packed, absmax, shape), w)
        e4 = Q2.rel(Q2.quant4bit_nf4(w), w)
        e8 = Q2.rel(Q2.quant8bit(w), w)
        sig = float(w.detach().float().norm(dim=1).sum())
        rows.append({"k": k, "comp": comp, "shape": tuple(w.shape), "n": w.numel(),
                     "e": {2: e2, 4: e4, 8: e8}, "sig": sig})
        if max_tensors and len(rows) >= max_tensors:
            break
    if fh is not None:
        fh.__exit__(None, None, None)
    if verbose:
        print(f"  measured {len(rows)} tensors in {time.time()-t0:.0f}s")
    return rows


def frontier(rows, weight_key):
    """[(size_bytes, weighted_error, assignment)] for every breakpoint of the greedy.

    Starts with everything at 2 bits and repeatedly applies the cheapest error
    reduction per byte, recording the state after each step. Every recorded state is a
    valid assignment, so the result is a real frontier rather than a fitted curve.
    """
    n = len(rows)
    cont = [float(r[weight_key]) for r in rows]
    tot_c = sum(cont)
    assign = [2] * n
    size = sum(r["n"] * bpw(2) for r in rows)

    def err(a):
        return sum(c * r["e"][b] for c, r, b in zip(cont, rows, a)) / max(1e-30, tot_c)

    states = [(size, err(assign), list(assign))]

    # candidate moves: 2->4 and 4->8, recomputed as we go
    for _ in range(2 * n):
        best = None
        for i, b in enumerate(assign):
            if b == 2:
                nb = 4
            elif b == 4:
                nb = 8
            else:
                continue
            d_err = cont[i] * (rows[i]["e"][b] - rows[i]["e"][nb]) / max(1e-30, tot_c)
            d_bytes = rows[i]["n"] * (bpw(nb) - bpw(b))
            if d_err <= 0 or d_bytes <= 0:
                continue
            score = d_err / d_bytes
            if best is None or score > best[0]:
                best = (score, i, nb, d_err, d_bytes)
        if best is None:
            break
        _, i, nb, d_err, d_bytes = best
        assign[i] = nb
        size += d_bytes
        states.append((size, err(assign), list(assign)))
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--max-tensors", type=int, default=48)
    ap.add_argument("--components", default="text_encoder,transformer")
    a = ap.parse_args()

    comps = []
    for c in a.components.split(","):
        c = c.strip()
        comps.append((c, "model" if c == "text_encoder" else
                      "diffusion_pytorch_model"))

    print(f"model: {a.model}")
    rows = measure(a.model, comps, a.max_tensors)
    if not rows:
        print("no readable tensors")
        return 2  # noqa: PLR1714  (kept explicit for readability)

    n_all = sum(r["n"] for r in rows)
    print()
    print("=" * 90)
    print("uniform schemes over the measured set")
    print("=" * 90)
    print(f"  {'bits':>4s} {'GB':>9s} {'weighted error':>15s} {'vs 4-bit':>10s}")
    base_size = base_err = None
    for b in BITS:
        size = sum(r["n"] * bpw(b) for r in rows) / 1e9
        err = sum(r["n"] * r["e"][b] for r in rows) / n_all
        if b == 4:
            base_size, base_err = size, err
        print(f"  {b:>4d} {size:9.4f} {err:15.6f}", end="")
        print(f" {100*err/base_err:9.1f}%" if base_err else "")
    print()

    for wkey, wlabel in (("n", "parameter count"), ("sig", "row-norm sum (signal proxy)")):
        states = frontier(rows, wkey)
        print("=" * 90)
        print(f"PARETO FRONTIER, weighted by {wlabel}")
        print("=" * 90)
        print(f"  {'size GB':>9s} {'error':>10s} {'vs all-4 err':>13s} "
              f"{'vs all-4 size':>14s}  {'2/4/8 counts':>14s}")
        for size, err, asg in states:
            c2 = asg.count(2)
            c4 = asg.count(4)
            c8 = asg.count(8)
            mark = ""
            if size / 1e9 <= base_size * 1.02 and err < base_err:
                mark = "  <-- beats all-4-bit: no bigger, less error"
            print(f"  {size/1e9:9.4f} {err:10.6f} {100*err/base_err:12.1f}% "
                  f"{100*size/1e9/base_size:13.1f}%  {c2:4d}/{c4:3d}/{c8:3d}{mark}")
        print()

    print("how to read this:")
    print("  each row is a real assignment, so a row that is smaller AND has less error")
    print("  than the all-4-bit line is a strict improvement. If no row manages that,")
    print("  mixing does not beat uniform 4-bit on this evidence, and saying so is the")
    print("  result -- not a failure to find the right mix.")
    print()
    print("still outstanding, and it decides whether any of this reaches a picture:")
    print("  the weights of one layer were measured to carry a relative error of 1.04")
    print("  while its OUTPUT error was 2.2e-05. Weight error and image quality are not")
    print("  proportional, so per-layer output sensitivity has to be measured on the")
    print("  loaded model before an allocation can be called good rather than smaller.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
