"""
Divergence forensics for the cross-machine 1000-step training comparison.

Question being answered
-----------------------
R5 and i5 produce loss trajectories that agree to ~7e-5 mean / 6.7e-4 max but are
NOT bit-identical.  Two competing explanations:

  (A) A kernel bug: some CPU kernel (gemm_8bit / gemv_4bit / quantize / gdn / moe)
      computes something systematically wrong, so the two machines optimise
      different functions.
  (B) Floating-point summation order: the two machines run different torch builds
      (2.13.0+cpu vs 2.14.0+cpu) on different microarchitectures (Zen2 vs Skylake),
      so elementwise/reduction order differs in the last ulp.  Training is a
      chaotic iteration: a 1-ulp seed amplifies over 1000 steps.

These have opposite signatures:
  (A) predicts the gap appears LATE and GROWS steadily / jumps at a specific step.
  (B) predicts the gap is ~0 for the first steps, appears at a single step as
      ~1 ulp of the loss, and then amplifies smoothly (roughly exponentially
      while the loss is still falling, then saturating once the trajectory
      converges to the flat tail).

This script reads both loss arrays and reports:
  1. first step where the two differ
  2. the magnitude of the difference at that step (the seed)
  3. the amplification factor from seed to final gap
  4. a fit of log(gap) vs step to get the growth rate (Lyapunov-ish exponent)
  5. whether the difference is consistent with pure amplification of an
     O(1-ulp) seed, i.e. seed * exp(rate * n) >= observed

If (5) holds with a seed of order 1e-7 (fp32 eps on a loss of ~4-5), then the
observed 6.7e-4 is fully explained and no kernel defect is implied.  A threshold
that demands tighter agreement than (2) over 1000 steps would also fail two
bit-identical kernels running on different summation orders.

Usage:
    python divergence_forensics.py r5.json i5.json
"""
import json
import math
import sys


def load(path):
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    return obj["losses"], obj


def fit_log_growth(steps, gaps):
    """Least-squares slope of log(gap) vs step over the segment where the gap is
    still in its growth phase (gap > 0 and before it saturates)."""
    xs, ys = [], []
    for s, g in zip(steps, gaps):
        if g > 0.0:
            xs.append(s)
            ys.append(math.log(g))
    if len(xs) < 3:
        return None, None, 0
    # Stop at the point of maximum gap: after that the trajectory has converged
    # and the gap stops growing, so it is not part of the growth phase.
    gmax = max(gaps)
    imax = gaps.index(gmax)
    xs = [s for s in xs if s <= imax]
    ys = ys[: len(xs)]
    n = len(xs)
    if n < 3:
        return None, None, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    return slope, my - slope * mx, n


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, oa = load(sys.argv[1])
    b, ob = load(sys.argv[2])
    n = min(len(a), len(b))

    gaps = [abs(a[i] - b[i]) for i in range(n)]
    steps = list(range(1, n + 1))

    print("=" * 72)
    print("DIVERGENCE FORENSICS")
    print("=" * 72)
    print(f"  run A : {oa.get('host', '?')}  torch {oa.get('torch', '?')}  py {oa.get('python', '?')}")
    print(f"  run B : {ob.get('host', '?')}  torch {ob.get('torch', '?')}  py {ob.get('python', '?')}")
    print(f"  steps compared: {n}")

    identical = sum(1 for g in gaps if g == 0.0)
    print()
    print(f"  bit-identical steps : {identical}/{n}")
    if identical == n:
        print("  RESULT: the two runs are bit-identical. Nothing to explain.")
        return 0

    # 1. first divergence
    first = next(i for i, g in enumerate(gaps) if g > 0.0)
    seed = gaps[first]
    print(f"  first divergence at step {first + 1}")
    print(f"    loss A = {a[first]!r}")
    print(f"    loss B = {b[first]!r}")
    print(f"    gap    = {seed:.3e}   (seed)")

    # how many steps stay identical around the seed
    run_ident = 0
    for i in range(first):
        if gaps[i] == 0.0:
            run_ident += 1
    print(f"    identical for the preceding {run_ident} steps")

    peak = max(abs(v) for v in a[:n])
    eps32 = 2.0 ** -23
    ulp = peak * eps32
    print()
    print(f"  loss peak           : {peak:.6f}")
    print(f"  1 ulp at that scale : {ulp:.3e}  (fp32 eps = {eps32:.3e})")
    print(f"  seed / ulp          : {seed / ulp:.3f}")

    # 2. amplification
    gmax = max(gaps)
    imax = gaps.index(gmax)
    print()
    print(f"  max gap             : {gmax:.3e}  at step {imax + 1}")
    print(f"  amplification seed->max : {gmax / seed:,.0f}x")

    # 3. growth rate fit
    slope, intercept, used = fit_log_growth(steps, gaps)
    print()
    if slope is None:
        print("  growth fit          : not enough growing samples")
    else:
        print(f"  log-gap growth fit over {used} steps (up to the max)")
        print(f"    slope = {slope:+.5f} per step   =>  e^{slope:.5f} = {math.exp(slope):.5f}x per step")
        per100 = math.exp(slope * 100)
        print(f"    over 100 steps: {per100:,.1f}x")
        predicted = seed * math.exp(slope * imax)
        print(f"    predicted max from seed alone: {predicted:.3e}   observed: {gmax:.3e}")
        ratio = predicted / gmax
        print(f"    predicted/observed = {ratio:.3f}")

    # 4. verdict on the seed hypothesis: is the whole thing explained by
    #    amplifying a last-bit seed?
    print()
    print("-" * 72)
    if slope is not None and seed <= 4 * ulp:
        print("  VERDICT: consistent with (B) floating-point summation order.")
        print(f"    The seed is {seed / ulp:.2f} ulp of the loss scale - the smallest")
        print("    representable difference.  Everything after that is the chaotic")
        print("    amplification of that single last-bit difference.")
    elif slope is None:
        print("  VERDICT: inconclusive (no growth phase to fit).")
    else:
        print("  VERDICT: seed is larger than a few ulp - inspect the first")
        print("    diverging step's kernels directly.")

    # 5. show the shape: gap at deciles
    print()
    print("  gap at deciles of the run:")
    print("    step      gap          growth vs prev decile")
    prev = None
    for k in range(1, 11):
        i = min(n, k * n // 10) - 1
        g = gaps[i]
        if prev is None or prev == 0:
            gs = "        -"
        else:
            gs = f"{g / prev:>8.2f}x"
        print(f"    {i + 1:>5}   {g:.3e}   {gs}")
        prev = g
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
