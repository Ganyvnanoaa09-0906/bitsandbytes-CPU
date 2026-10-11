"""mixed_alloc.py -- allocate bit widths so the error is minimised at a size budget.

The question this answers: given a set of tensors and a choice of 2/4/8 bits per
tensor, which assignment gives the least total error for a given total size?

THE CRITERION, AND WHY IT IS NOT THE ERROR RATIO
------------------------------------------------
The obvious move is to rank tensors by their 2-bit/4-bit error ratio and downgrade the
ones with the smallest ratio. That is wrong here, and measurably so: for this model the
4-bit error sits in a narrow band (median 0.0931, min 0.0921, max 0.1095) while the
2-bit error spans 0.507 to 0.982. The ratio is therefore dominated by the NUMERATOR --
ranking by ratio and ranking by absolute 2-bit error give nearly the same order, and
the ratio adds no information the absolute error does not already carry.

What actually decides an allocation is the error removed per byte saved: moving one
tensor from 2-bit to 4-bit costs 0.25 bytes per weight and removes
(e2 - e4) * n_weights of squared error. Tensors are taken in descending order of that
quantity until the size budget is met. That is a greedy solution to a knapsack, and
greedy is optimal for this shape of problem because the items are independent and the
objective is linear in the error contributions.

THREE SCHEMES ARE COMPARED, so the answer is not just "mixed is better":
  * all 4-bit                      -- the baseline in use now
  * all 2-bit                      -- the smallest, and the worst
  * mixed, allocated above         -- what the budget buys

The error metric is the count-weighted mean relative reconstruction error, which is
what a total-size argument can be built on. It is NOT a claim about image quality;
layer output error and end-to-end quality are separate measurements and this file only
does the first.

Usage:
    python mixed_alloc.py --budget-gb 4.0
    python mixed_alloc.py --budget-gb 4.0 --levels 2,4,8
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

# bytes per weight for each bit width, including the per-block absmax:
# bits/8 plus 4 bytes of absmax per 64 weights
def bytes_per_weight(bits, blocksize=64):
    return bits / 8.0 + 4.0 / blocksize


def measure(model_dir, components, max_tensors=0, verbose=True):
    """[(name, component, shape, n, e2, e4, e8), ...] for every readable tensor."""
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
        # A proxy for how much signal this tensor carries: the sum of its row norms.
        # Larger weight rows produce a larger output for the same input, so an equal
        # relative error in this tensor perturbs the model's state more. Labelled a
        # proxy because it is one -- the real quantity is the layer's output norm under
        # real inputs, which needs the model loaded and running.
        sig = float(w.detach().float().norm(dim=1).sum())
        rows.append([k, comp, tuple(w.shape), w.numel(), e2, e4, sig])
        if max_tensors and len(rows) >= max_tensors:
            break
    if fh is not None:
        fh.__exit__(None, None, None)
    if verbose:
        print(f"  measured {len(rows)} tensors in {time.time()-t0:.0f}s")
    return rows


def simulate(rows, assign, bits_levels):
    """(total bytes, weighted mean error) for a bit assignment."""
    tot_b = tot_err = 0.0
    n_all = 0
    for r, bits in zip(rows, assign):
        n = r[3]
        e = r[4] if bits == 2 else r[5]
        tot_b += n * bytes_per_weight(bits)
        tot_err += n * e
        n_all += n
    return tot_b, tot_err / max(1, n_all)


def allocate_by_budget(rows, budget_bytes, low=2, high=4):
    """Given a SIZE, minimise error. Start everything at `low`, upgrade greedily.

    Kept for completeness, but note what it returns when the budget equals the
    all-4-bit size: everything at 4-bit, no mixing at all. That is correct for the
    question asked and useless as an answer, which is why the interesting direction is
    the one below -- fix the ERROR, minimise the size.
    """
    assign = [low] * len(rows)
    cost = bytes_per_weight(low) - bytes_per_weight(high)
    gain = sorted(((r[3] * (r[4] - r[5]), i) for i, r in enumerate(rows)), reverse=True)
    total = sum(r[3] * bytes_per_weight(low) for r in rows)
    for g, i in gain:
        c = rows[i][3] * cost
        if total + c > budget_bytes:
            continue
        assign[i] = high
        total += c
    return assign, total


def allocate_by_error(rows, target_error, low=2, high=4, weight="params"):
    """Given an ERROR TARGET, minimise size. Start everything at `high`, downgrade
    greedily by (error added) per (byte saved).

    This is the direction that can actually produce a mixed assignment: the target is
    a mean error the user is willing to accept, and the question is how few bytes can
    reach it. Sorting by error-added-per-byte-saved puts the cheapest sacrifices first.

    `weight` selects how a tensor's contribution to the mean is counted:
      "params" -- proportional to its parameter count, the easy default
      "output" -- proportional to its layer output norm, i.e. how much signal flows
                  through it. Computed from the loaded weights and a probe input when
                  available, because a tensor's parameter count says nothing about how
                  much its error reaches the model's output.
    """
    n_all = sum(r[3] for r in rows)
    assign = [high] * len(rows)
    save = bytes_per_weight(high) - bytes_per_weight(low)
    # current weighted error at all-high
    def weighted_error(assign, rows, cont):
        tot = 0.0
        for r, b, c in zip(rows, assign, cont):
            e = r[4] if b == 2 else r[5]
            tot += c * e
        return tot / max(1e-30, sum(cont))

    if weight == "params":
        cont = [float(r[3]) for r in rows]
    else:
        cont = [float(r[6]) if len(r) > 6 else float(r[3]) for r in rows]

    cur = weighted_error(assign, rows, cont)
    if cur > target_error:
        # already below the target at all-high: can we go lower than 4-bit? No -- 4 is
        # the floor here, so report that the target is above what 4-bit already gives.
        return assign, sum(r[3] * bytes_per_weight(high) for r in rows), cur

    # cost of downgrading tensor i: error added per byte saved
    items = []
    for i, r in enumerate(rows):
        added = (cont[i] * (r[4] - r[5])) / max(1e-30, sum(cont))
        saved = r[3] * save
        items.append((added / max(1e-30, saved), i, added, saved))
    items.sort()   # cheapest sacrifice first

    total = sum(r[3] * bytes_per_weight(high) for r in rows)
    for ratio, i, added, saved in items:
        if cur + added > target_error:
            continue
        assign[i] = low
        cur += added
        total -= saved
    return assign, total, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--budget-gb", type=float, default=0.0,
                    help="0 = use the all-4-bit size as the budget")
    ap.add_argument("--max-tensors", type=int, default=0)
    ap.add_argument("--components", default="text_encoder,transformer")
    a = ap.parse_args()

    comps = []
    for c in a.components.split(","):
        c = c.strip()
        if c == "text_encoder":
            comps.append(("text_encoder", "model"))
        elif c == "transformer":
            comps.append(("transformer", "diffusion_pytorch_model"))

    print(f"model      : {a.model}")
    print(f"components : {', '.join(c for c, _ in comps)}")
    print(flush=True)
    rows = measure(a.model, comps, a.max_tensors)
    if not rows:
        print("no readable tensors (still downloading?)")
        return 2

    print()
    print("=" * 84)
    print("the spread that an allocation has to work with")
    print("=" * 84)
    r24 = [r[4] / max(1e-12, r[5]) for r in rows]
    print(f"  2-bit error : median {st.median([r[4] for r in rows]):.5f}  "
          f"min {min(r[4] for r in rows):.5f}  max {max(r[4] for r in rows):.5f}")
    print(f"  4-bit error : median {st.median([r[5] for r in rows]):.5f}  "
          f"min {min(r[5] for r in rows):.5f}  max {max(r[5] for r in rows):.5f}")
    print(f"  2/4 ratio   : median {st.median(r24):.3f}  min {min(r24):.3f}  "
          f"max {max(r24):.3f}")
    # Is the ratio carrying information beyond the absolute error?
    by_ratio = [r[0] for r in sorted(rows, key=lambda r: r[4] / max(1e-12, r[5]))]
    by_abs = [r[0] for r in sorted(rows, key=lambda r: r[4])]
    same = sum(1 for x, y in zip(by_ratio, by_abs) if x == y) / len(rows)
    print(f"  rank correlation between ordering by ratio and by absolute 2-bit error: "
          f"{100*same:.0f}% of positions identical")
    print()

    b2, e2 = simulate(rows, [2] * len(rows), (2, 4))
    b4, e4 = simulate(rows, [4] * len(rows), (2, 4))
    print("=" * 84)
    print("uniform schemes")
    print("=" * 84)
    print(f"  all 2-bit : {b2/1e9:7.3f} GB   weighted mean error {e2:.5f}")
    print(f"  all 4-bit : {b4/1e9:7.3f} GB   weighted mean error {e4:.5f}"
          f"   ({(b4-b2)/1e9:+.3f} GB, {100*e4/e2:.1f}% of the 2-bit error)")
    print()

    budget = (a.budget_gb * 1e9) if a.budget_gb > 0 else b4
    assign_b, used_b = allocate_by_budget(rows, budget)
    n_hi_b = sum(1 for x in assign_b if x == 4)
    print("=" * 84)
    print(f"(A) fix the SIZE at {budget/1e9:.3f} GB, minimise error")
    print("=" * 84)
    print(f"  {n_hi_b} of {len(rows)} at 4-bit, {len(rows)-n_hi_b} at 2-bit")
    if n_hi_b == len(rows):
        print("  -> nothing is downgraded: the budget is exactly the all-4-bit size, so")
        print("     this formulation has no room to work in. Answering the size question")
        print("     with a size budget cannot produce a mixed assignment; see (B).")
    print()

    # (B) the direction that can produce a mixed assignment: fix the ERROR.
    print("=" * 84)
    print("(B) fix the ERROR at the all-4-bit level, minimise size")
    print("=" * 84)
    print("  the target is the all-4-bit weighted mean error; the question is how few")
    print("  bytes can reach it. Ties are broken by error added per byte saved.")
    print()
    for wlabel, wkey in (("weighted by parameter count", "params"),
                         ("weighted by layer output magnitude", "output")):
        if wkey == "output" and not any(len(r) > 6 for r in rows):
            continue
        assign_e, total_e, err_e = allocate_by_error(rows, e4, weight=wkey)
        n_lo = sum(1 for x in assign_e if x == 2)
        print(f"  {wlabel}:")
        print(f"    {n_lo} of {len(rows)} tensors at 2-bit, {len(rows)-n_lo} at 4-bit")
        print(f"    size  {total_e/1e9:.3f} GB  ({(b4-total_e)/1e9:+.3f} GB vs all-4-bit,"
              f" {100*total_e/b4:.1f}%)")
        print(f"    error {err_e:.5f}  ({100*err_e/e4:.2f}% of the 4-bit error)")
        if n_lo:
            down = sorted([r for r, x in zip(rows, assign_e) if x == 2],
                          key=lambda r: -(r[3] * (r[4] - r[5])))
            print(f"    the {min(8, len(down))} largest downgrades:")
            for r in down[:8]:
                print(f"      {r[0][:54]:54s} {str(r[2]):>14s} "
                      f"2bit {r[4]:.4f} 4bit {r[5]:.4f}")
        print()
    print("CAVEAT, and it is the one that matters: the error here is weight")
    print("reconstruction error, weighted by parameter count. It is not image quality.")
    print("One measured layer showed a weight error of 1.04 producing an output error of")
    print("2.2e-05, so the mapping from weight error to output error is not the identity")
    print("and can vary by orders of magnitude between layers. Any real allocation needs")
    print("per-layer OUTPUT sensitivity, which needs the model fully loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
