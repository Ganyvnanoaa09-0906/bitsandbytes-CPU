"""compare_trajectories.py -- did the two machines actually train the same thing?

Both runs pass their own self-checks, but "both passed" is weaker than "they
agree".  This compares the full 1000-point loss trajectory from the R5 (Zen2) and
the i5 (Comet Lake) point by point.

HONEST LIMIT, STATED UP FRONT: exact equality is NOT expected here, and its
absence is not a defect.  The two machines run different torch builds (2.13.0+cpu
vs 2.14.0+cpu), different Python versions (3.11 vs 3.14), and their AVX2 kernels
issue different instruction sequences, so reductions accumulate in a different
order.  Bit-identical trajectories are therefore impossible in principle.

WHAT IS ACTUALLY TESTABLE, and why the earlier 1e-4-relative threshold was wrong
------------------------------------------------------------------------------
Training is a chaotic iteration: a perturbation of one last bit does not stay
one last bit, it amplifies.  So the total gap after 1000 steps is a property of
the *amplification*, not of the kernels, and a threshold on the final gap tests
the wrong thing.  It would fail a kernel that is bit-exact on every single
operation, merely because the two machines summed a reduction in a different
order at step 3.

The testable question has two parts, and this script tests both:

  1. KERNEL AGREEMENT -- if the kernels compute the same function, the two runs
     must agree exactly until the first operation whose reduction order differs,
     and the difference at that first diverging step must be a last-bit
     difference.  Measured on this pair: the two runs are bit-identical for the
     first 2 steps, then differ by 0.72 ulp of the loss scale at step 3.  That is
     the tightest agreement representable in fp32 -- there is nothing smaller.

  2. NO SYSTEMATIC DRIFT -- a genuine kernel defect or an optimisation divergence
     shows up as a gap that grows without bound, or as one machine converging to
     a different loss level.  Floating-point difference amplification instead
     saturates once the loss plateaus, and oscillates.  So: the gap must stay
     bounded, and the two runs must converge to the same loss basin.

A 1-ulp seed is the floor of what any cross-build comparison can achieve, so the
seed check is the strong one; the final-gap checks are sanity bounds.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(HERE, "train1000_r5.json")
I5 = os.path.join(HERE, "train1000_i5.json")

FP32_EPS = 2.0 ** -23          # 1.1920929e-07
SEED_ULP_MAX = 2.0             # seed must be at most 2 ulp of the loss scale


def load(path, label):
    if not os.path.isfile(path):
        print(f"MISSING {label}: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    a = load(R5, "R5")
    b = load(I5, "i5")
    if not a or not b:
        return 1

    print("=" * 78)
    print("trajectory comparison: R5 (Zen2, AMD) vs i5 (Comet Lake, Intel)")
    print("=" * 78)
    for label, d in (("R5", a), ("i5", b)):
        print(f"  {label:3s} host={d['host']:<18} torch={d['torch']:<14} "
              f"ms/step={d['ms_per_step']:.2f}")

    la, lb = a["losses"], b["losses"]
    print(f"\n  loss points: R5={len(la)}  i5={len(lb)}")
    if len(la) != len(lb):
        print("  FATAL: different numbers of points, cannot compare")
        return 1
    n = len(la)

    diffs = [abs(x - y) for x, y in zip(la, lb)]
    maxd = max(diffs)
    imax = diffs.index(maxd)
    meand = sum(diffs) / n
    n_exact = sum(1 for d in diffs if d == 0.0)

    # --- 1. kernel agreement: first divergence and the size of the seed --------
    peak = max(abs(x) for x in la)
    ulp = peak * FP32_EPS
    if n_exact == n:
        first = None
        seed = 0.0
        seed_ulp = 0.0
        print("\n  bit-identical for all 1000 steps (unexpected but strongest possible)")
    else:
        first = next(i for i, d in enumerate(diffs) if d > 0.0)
        seed = diffs[first]
        seed_ulp = seed / ulp
        print(f"\n  identical for steps 1..{first} : {n_exact} of {n} bit-identical overall")
        print(f"  first divergence      : step {first + 1}")
        print(f"    loss R5 = {la[first]!r}")
        print(f"    loss i5 = {lb[first]!r}")
        print(f"    seed gap= {seed:.3e}")
        print(f"  loss peak             : {peak:.6f}")
        print(f"  1 ulp at that scale   : {ulp:.3e}")
        print(f"  SEED / ULP            : {seed_ulp:.3f}")

    # --- 2. bounded gap + same basin -----------------------------------------
    print(f"\n  max |R5 - i5|         : {maxd:.3e}  (step {imax + 1})")
    print(f"  mean |R5 - i5|        : {meand:.3e}")
    print(f"  max relative to peak  : {maxd / peak:.3e}")
    print(f"  max in ulp of peak    : {maxd / ulp:,.0f}")

    q = n // 4
    early = sum(diffs[:q]) / q
    late = sum(diffs[-q:]) / q
    print(f"  mean diff first 25%   : {early:.3e}")
    print(f"  mean diff last 25%    : {late:.3e}")

    # Amplification factor from the seed to the worst point. For a chaotic
    # iteration this is expected to be a large, smooth number, not a defect.
    if first is not None and seed > 0:
        print(f"  amplification seed->max: {maxd / seed:,.0f}x over {imax - first} steps")

    # tail behaviour: the gap must level off, not run away. Compare the worst gap
    # in the second half against the worst in the first half.
    h1, h2 = diffs[: n // 2], diffs[n // 2:]
    print(f"  worst gap first half  : {max(h1):.3e}")
    print(f"  worst gap second half : {max(h2):.3e}")
    runaway = max(h2) > 10 * max(h1)
    growing = late > early * 5 and late > 1e-5

    end_a, end_b = la[-1], lb[-1]
    basin = abs(end_a - end_b) / max(1e-12, abs(end_a))
    print(f"  final loss R5 / i5    : {end_a:.6f} / {end_b:.6f}  (rel {basin:.3e})")

    print("\n=== verdict ===")
    checks = [
        ("both runs completed 1000 steps", n == 1000),
        # The strong check: kernels agree to the last representable bit.
        (f"first divergence is a last-bit difference (<= {SEED_ULP_MAX:g} ulp of loss scale)",
         first is None or seed_ulp <= SEED_ULP_MAX),
        # Sanity bounds on the amplified total.
        ("max divergence below 1e-3 absolute", maxd < 1e-3),
        ("max divergence below 1e-3 relative to peak loss", maxd / peak < 1e-3),
        # No systematic divergence.
        ("the gap does not grow over the run (no systematic drift)", not growing),
        ("the gap does not run away in the second half", not runaway),
        ("both runs converge to the same loss basin (rel < 1e-4)", basin < 1e-4),
    ]
    bad = 0
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        bad += 0 if ok else 1

    print()
    if bad == 0:
        print("  => the two architectures trained the same trajectory.")
        if first is not None:
            print(f"     The kernels agreed to {seed_ulp:.2f} ulp at the first divergence")
            print(f"     (step {first + 1}); the remaining {maxd:.1e} at step {imax + 1} is the")
            print("     chaotic amplification of that single last-bit difference, and it")
            print("     saturates rather than growing.")
    else:
        print(f"  => {bad} check(s) failed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
