# -*- coding: utf-8 -*-
"""Temporal consistency metrics for generated video frames.

Quantifies the "时序崩溃" failure mode so base-model / adapter / scheduler choices can be
compared objectively instead of by eye.

Metrics per clip:
  mad        mean absolute frame-to-frame difference (0-255). Lower = smoother.
  mad_first  MAD between frame0 and frame1 (the known "first-frame jump").
  warp_err   mean abs diff after aligning frame t+1 to t by a global translational shift
             (cheap proxy for motion-compensated error; large drop vs mad means the change
             is mostly camera/global motion rather than content collapse).
  hist_drift mean L1 distance between consecutive colour histograms (appearance drift).
  sat_mean   mean saturation (anime art is typically more saturated).

Usage:
  python temporal_metrics.py <glob-or-dir> [more...]
  python temporal_metrics.py "D:/work/output/adiff_out/anime512*.png"
"""
import sys
import os
import glob
import numpy as np
from PIL import Image


def load_frames(paths):
    return [np.asarray(Image.open(p).convert("RGB")).astype(np.float32) for p in paths]


def best_shift_err(a, b, rng=6):
    """Mean abs diff between a and b after shifting b by (dy,dx) in [-rng, rng]."""
    h, w = a.shape[:2]
    best = None
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            y0a, y1a = max(0, dy), min(h, h + dy)
            x0a, x1a = max(0, dx), min(w, w + dx)
            y0b, y1b = max(0, -dy), min(h, h - dy)
            x0b, x1b = max(0, -dx), min(w, w - dx)
            if y1a <= y0a or x1a <= x0a:
                continue
            d = np.abs(a[y0a:y1a, x0a:x1a] - b[y0b:y1b, x0b:x1b]).mean()
            if best is None or d < best:
                best = d
    return best if best is not None else float("nan")


def hist_l1(a, b, bins=32):
    out = 0.0
    for c in range(3):
        ha, _ = np.histogram(a[:, :, c], bins=bins, range=(0, 255), density=True)
        hb, _ = np.histogram(b[:, :, c], bins=bins, range=(0, 255), density=True)
        out += np.abs(ha - hb).sum()
    return out / 3.0


def saturation(a):
    mx = a.max(2)
    mn = a.min(2)
    return float(((mx - mn) / (mx + 1e-6)).mean())


def white_ratio(a, thresh=245):
    """Fraction of near-white pixels (%).

    This is the metric that exposed the sliding-window whitening drift: MAD and histogram
    L1 both stayed flat while highlights slowly blew out (0.07% -> 9.4% over 34 frames).
    A global brightening shifts every pixel a little, which MAD under-weights, and the
    histogram L1 distance gets diluted by the many unchanged pixels.
    """
    return float(100.0 * (a > thresh).mean())


def p99(a):
    return float(np.percentile(a, 99))


def lapvar(a):
    """Laplacian variance on luma: proxy for high-frequency detail energy.

    Quantifies the "frosted glass" detail melt. The old (timestep-mismatched) 512 sliding
    window collapsed 423.8 -> 26.8 across 34 frames, a 94% loss. Always cast to float first:
    integer subtraction underflows and silently corrupts the result.
    """
    a = a.astype(np.float32)
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def analyze(paths, name, chunk=None, overlap=None):
    fr = load_frames(paths)
    mads, warps, hists = [], [], []
    for i in range(len(fr) - 1):
        mads.append(float(np.abs(fr[i + 1] - fr[i]).mean()))
        warps.append(float(best_shift_err(fr[i], fr[i + 1])))
        hists.append(float(hist_l1(fr[i], fr[i + 1])))
    sats = [saturation(f) for f in fr]
    print(f"=== {name}  ({len(fr)} frames) ===")
    print(f"  mad        mean {np.mean(mads):6.2f}   min {np.min(mads):6.2f}   max {np.max(mads):6.2f}")
    print(f"  mad_first  {mads[0]:6.2f}   (frame0->frame1)")
    print(f"  warp_err   mean {np.mean(warps):6.2f}   (motion-compensated)")
    print(f"  residual   {np.mean(mads) - np.mean(warps):6.2f}   (mad - warp_err; high = content change)")
    print(f"  hist_drift mean {np.mean(hists):6.4f}")
    print(f"  sat_mean   {np.mean(sats):6.3f}")

    # ---- whitening / highlight-blowout check (the metric that caught the drift) ----
    wr = [white_ratio(f) for f in fr]
    p99s = [p99(f) for f in fr]
    means = [float(f.mean()) for f in fr]
    print(f"  white%>245 first {wr[0]:6.2f}  last {wr[-1]:6.2f}  "
          f"ratio {wr[-1]/max(wr[0],1e-6):8.2f}x")
    print(f"  p99        first {p99s[0]:6.1f}  last {p99s[-1]:6.1f}")
    print(f"  frame mean first {means[0]:6.1f}  last {means[-1]:6.1f}  "
          f"({means[-1]-means[0]:+.1f})")
    # per-frame trend of white%: a monotone climb means highlights are accumulating
    if len(wr) >= 4:
        idx = np.arange(len(wr))
        slope = float(np.polyfit(idx, wr, 1)[0])
        monotone = sum(1 for i in range(len(wr) - 1) if wr[i + 1] >= wr[i] - 0.05)
        verdict = ("WHITENING DRIFT (highlights accumulating)"
                   if slope > 0.05 and monotone > 0.7 * (len(wr) - 1)
                   else "stable")
        print(f"  white% slope {slope:+.4f}/frame, monotone steps {monotone}/{len(wr)-1}"
              f"  -> {verdict}")

    # ---- detail-energy check (the "frosted glass" metric) ----
    laps = [lapvar(f) for f in fr]
    if laps[0] > 0:
        print(f"  Laplacian var first {laps[0]:8.1f}  last {laps[-1]:8.1f}  "
              f"({(laps[-1]-laps[0])/laps[0]*100:+.1f}%)")
        if chunk:
            per = [float(np.mean(laps[i:i + chunk])) for i in range(0, len(laps), chunk)]
            print(f"  Laplacian var per chunk: " + " ".join(f"{v:.1f}" for v in per))
        drop = (laps[0] - laps[-1]) / laps[0] * 100
        if drop > 80:
            print(f"  -> DETAIL MELT ({drop:.0f}% loss): frosted-glass territory")
        elif drop > 40:
            print(f"  -> detail softens {drop:.0f}% (2-step relay cost, 'anime soft-focus' range)")
        else:
            print(f"  -> detail holds ({drop:.0f}% loss)")

    print(f"  per-step mad: " + " ".join(f"{m:.1f}" for m in mads))

    # If this is a sliding-window clip, split transitions into intra-chunk vs at-boundary.
    if chunk and overlap:
        stride = chunk - overlap
        bounds = set(range(stride, len(fr) - 1, stride))  # transition index i means fr[i]->fr[i+1]
        intra = [mads[i] for i in range(len(mads)) if i not in bounds]
        at_b = [mads[i] for i in range(len(mads)) if i in bounds]
        print(f"  -- boundary split (chunk={chunk} overlap={overlap} stride={stride}) --")
        if intra:
            print(f"  intra-chunk mad : {np.mean(intra):6.2f}  (n={len(intra)})")
        if at_b:
            print(f"  AT-BOUNDARY mad : {np.mean(at_b):6.2f}  (n={len(at_b)})  "
                  f"indices={sorted(bounds)}")
        if intra and at_b:
            r = np.mean(at_b) / max(np.mean(intra), 1e-9)
            print(f"  boundary/intra ratio: {r:.2f}x  "
                  f"({'seam visible' if r > 1.5 else 'seam well hidden'})")

    return dict(mad=float(np.mean(mads)), mad_first=mads[0], warp=float(np.mean(warps)),
                resid=float(np.mean(mads) - np.mean(warps)), hist=float(np.mean(hists)),
                sat=float(np.mean(sats)))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    groups = {}
    for a in args:
        if os.path.isdir(a):
            for p in sorted(glob.glob(os.path.join(a, "*.png"))):
                groups.setdefault(os.path.basename(a), []).append(p)
        else:
            for p in sorted(glob.glob(a)):
                # group by stripping the trailing _fN.png
                base = os.path.basename(p)
                import re
                key = re.sub(r"_f\d+\.png$", "", base)
                groups.setdefault(key, []).append(p)
    for k, v in groups.items():
        if len(v) >= 2:
            # sort by the frame index embedded in the filename so MAD is over real order
            import re
            v = sorted(v, key=lambda p: int(re.search(r"_f(\d+)\.png$", p).group(1)))
            analyze(v, k)
        else:
            print(f"(skip {k}: only {len(v)} frame(s))")


if __name__ == "__main__":
    main()
