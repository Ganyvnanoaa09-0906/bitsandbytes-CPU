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

  # E2E 管线（统一 metrics.json）：
  python temporal_metrics.py runs/<id>/frames --chunk 8 --overlap 0 \
      --json-out runs/<id>/metrics.json --run-id <id>

E2E-PATCH（2026-10-23，feat/video-e2e；port 自已验证的 E2E 副本）：
  [P1] 新增 `--json-out`：输出统一 `video_e2e/1` schema（帧/时序/white/detail/boundary/verdict）。
  [P2] 新增 `best_shift()`（返回 err,dx,dy）；**`best_shift_err()` 签名保持不变**（向后兼容）。
  [P3] 新增 `--chunk/--overlap/--rng/--run-id/--extra-json` CLI；修边界拆分的触发条件
       （原 `if chunk and overlap:` 在 overlap=0 的独立分块模式静默跳过）。
  [P3b] 修边界索引差一：原 `range(stride, …)` 与 independent_chunks 自己的 seam 口径
       （`(i+1)%chunk==0`）不一致；改为 `range(chunk-1, …)`。
  [P4] 新增 `has_global_motion`（warp/mad<0.85 且存在非零众数位移才判"有整体运动"）。
  [P5] 修 white% 漂移判据：原 `slope>0.05 and monotone>0.7` 会把"大量 0 + 尖峰"误报为漂移
       （实测 R² 仅 0.17–0.24）；改为 `slope>0.05 and R²>0.5`，另把峰值>10% 记为 hot-VAE warn。
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys
import os
import glob
import json
import numpy as np
from PIL import Image

SCHEMA = "video_e2e/1"


def _reconfigure_stdout():
    # 本机 PowerShell 子进程 stdout 是 GBK；输出含 '⇒' 等字符会抛 UnicodeEncodeError。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_frames(paths):
    return [np.asarray(Image.open(p).convert("RGB")).astype(np.float32) for p in paths]


def best_shift_err(a, b, rng=6):
    """Mean abs diff between a and b after shifting b by (dy,dx) in [-rng, rng].

    NOTE: 原签名/返回值保持不变（其他脚本 import 它）；要拿位移请用 best_shift()。
    """
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


def best_shift(a, b, rng=6):
    """E2E-PATCH[P2]: 返回 (err, dx, dy)；err 与 best_shift_err 完全一致。"""
    h, w = a.shape[:2]
    best = None
    best_dx = best_dy = 0
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
                best_dx, best_dy = dx, dy
    if best is None:
        return float("nan"), 0, 0
    return float(best), int(best_dx), int(best_dy)


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


def _white_stats(wr, n):
    """E2E-PATCH[P5]: white% 趋势统计（slope + R² + 峰值），返回 dict 供 analyze/analyze_sequence 共用。"""
    st = dict(first=float(wr[0]), last=float(wr[-1]), slope_per_frame=0.0, r2=0.0,
              monotone_steps=0, max=float(np.max(wr)),
              per_frame=[round(float(x), 3) for x in wr], verdict="stable")
    if n >= 4:
        idx = np.arange(n)
        k, b = np.polyfit(idx, wr, 1)
        yhat = k * idx + b
        ss_res = float(((np.array(wr) - yhat) ** 2).sum())
        ss_tot = float(((np.array(wr) - np.mean(wr)) ** 2).sum())
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        monotone = sum(1 for i in range(n - 1) if wr[i + 1] >= wr[i] - 0.05)
        # E2E-PATCH[P5]：只有线性趋势成立（R²>0.5）才叫 whitening drift
        st.update(slope_per_frame=float(k), r2=float(r2),
                  monotone_steps=f"{monotone}/{n-1}",
                  verdict=("WHITENING DRIFT (highlights accumulating)"
                           if (k > 0.05 and r2 > 0.5) else "stable"))
    return st


def _boundary_stats(mads, chunk, overlap):
    """E2E-PATCH[P3] + E2E-PATCH[P3b]: intra-chunk vs at-boundary MAD（overlap=0 也要算）。

    [P3] 触发条件：原实现 `if chunk and overlap:` 在独立分块模式（overlap=0）静默跳过。
    [P3b] 索引差一：原 `range(stride, …)` 与 independent_chunks.py 的 seam 口径不一致，
          改为 `range(chunk-1, …)`。
    """
    if chunk is None:
        return None
    stride = chunk - (overlap or 0)
    if stride <= 0:
        stride = chunk
    # chunk0 覆盖 0..chunk-1，第一个 seam 在 transition index chunk-1（fr[chunk-1]->fr[chunk]）；
    # 与 independent_chunks.py 的 seam 口径 `(i+1)%chunk==0` 一致（i = chunk-1 + k*stride）。
    bounds = set(range(chunk - 1, len(mads), stride))
    intra = [mads[i] for i in range(len(mads)) if i not in bounds]
    at_b = [mads[i] for i in range(len(mads)) if i in bounds]
    ratio = (float(np.mean(at_b) / max(np.mean(intra), 1e-9))
             if (intra and at_b) else None)
    return dict(mode="sliding" if (overlap or 0) > 0 else "independent",
                chunk=int(chunk), overlap=int(overlap or 0), stride=int(stride),
                intra_mad=float(np.mean(intra)) if intra else None,
                boundary_mad=float(np.mean(at_b)) if at_b else None,
                ratio=ratio, indices=sorted(bounds),
                note=("independent 模式 boundary 高 = 硬切，属预期，不是缺陷"
                      if (overlap or 0) == 0 else
                      "sliding 模式要求 ratio<=1.5（seam 藏住）"))


def analyze(paths, name, chunk=None, overlap=None):
    """原接口保持不变（print + 返回 dict；新字段只增不改）。"""
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
    white = _white_stats(wr, len(wr))
    if len(wr) >= 4:
        print(f"  white% slope {white['slope_per_frame']:+.4f}/frame, "
              f"R2={white['r2']:.3f}, monotone {white['monotone_steps']}"
              f"  -> {white['verdict']}")
        if white["max"] > 10.0 and white["verdict"] == "stable":
            print(f"  [warn] white% 峰值 {white['max']:.1f}% >10%（高光过曝，疑似 hot VAE："
                  f"设 ADIFF_VAE=plain SD1.5 VAE 可解，见 anime_adiff.py L24–28）")

    # ---- detail-energy check (the "frosted glass" metric) ----
    laps = [lapvar(f) for f in fr]
    detail = dict(lap_first=laps[0], lap_last=laps[-1], drop_pct=0.0, per_chunk=[])
    if laps[0] > 0:
        print(f"  Laplacian var first {laps[0]:8.1f}  last {laps[-1]:8.1f}  "
              f"({(laps[-1]-laps[0])/laps[0]*100:+.1f}%)")
        if chunk:
            per = [float(np.mean(laps[i:i + chunk])) for i in range(0, len(laps), chunk)]
            print(f"  Laplacian var per chunk: " + " ".join(f"{v:.1f}" for v in per))
            detail["per_chunk"] = per
        drop = (laps[0] - laps[-1]) / laps[0] * 100
        detail["drop_pct"] = drop
        if drop > 80:
            print(f"  -> DETAIL MELT ({drop:.0f}% loss): frosted-glass territory")
        elif drop > 40:
            print(f"  -> detail softens {drop:.0f}% (2-step relay cost, 'anime soft-focus' range)")
        else:
            print(f"  -> detail holds ({drop:.0f}% loss)")

    print(f"  per-step mad: " + " ".join(f"{m:.1f}" for m in mads))

    # If this is a sliding-window clip, split transitions into intra-chunk vs at-boundary.
    boundary = _boundary_stats(mads, chunk, overlap)
    if boundary is not None:
        print(f"  -- boundary split (chunk={chunk} overlap={overlap} stride={boundary['stride']}) --")
        if boundary["intra_mad"] is not None:
            n_intra = len(mads) - len(boundary["indices"])
            print(f"  intra-chunk mad : {boundary['intra_mad']:6.2f}  (n={n_intra})")
        if boundary["boundary_mad"] is not None:
            print(f"  AT-BOUNDARY mad : {boundary['boundary_mad']:6.2f}  "
                  f"(n={len(boundary['indices'])})  indices={boundary['indices']}")
        if boundary["ratio"] is not None:
            print(f"  boundary/intra ratio: {boundary['ratio']:.2f}x  "
                  f"({'seam visible' if boundary['ratio'] > 1.5 else 'seam well hidden'})")

    return dict(mad=float(np.mean(mads)), mad_first=mads[0], warp=float(np.mean(warps)),
                resid=float(np.mean(mads) - np.mean(warps)), hist=float(np.mean(hists)),
                sat=float(np.mean(sats)),
                # E2E-PATCH: 新增结构化字段（只增不改）
                white=white, detail=detail, boundary=boundary, per_step_mad=mads)


def analyze_sequence(paths, name, chunk=None, overlap=None, rng=6):
    """E2E-PATCH[P1]: 管线用结构化分析（不打印或少量打印），返回统一 schema 的 dict。"""
    fr = load_frames(paths)
    n = len(fr)
    mads, warps, hists, shifts = [], [], [], []
    for i in range(n - 1):
        mads.append(float(np.abs(fr[i + 1] - fr[i]).mean()))
        err, dx, dy = best_shift(fr[i], fr[i + 1], rng=rng)
        warps.append(err)
        shifts.append((dx, dy))
        hists.append(float(hist_l1(fr[i], fr[i + 1])))
    sats = [saturation(f) for f in fr]
    wr = [white_ratio(f) for f in fr]
    white = _white_stats(wr, n)
    # 与 E2E 副本 schema 对齐（数值一致；这些字段供 analyze/日志使用）
    p99s = [p99(f) for f in fr]
    means = [float(f.mean()) for f in fr]
    white.update(p99_first=p99s[0], p99_last=p99s[-1],
                 mean_first=means[0], mean_last=means[-1])
    if white["max"] > 10.0 and white["verdict"] == "stable":
        print(f"  [warn] white% 峰值 {white['max']:.1f}% >10%（高光过曝，疑似 hot VAE）")
    laps = [lapvar(f) for f in fr]
    detail = dict(lap_first=laps[0], lap_last=laps[-1], drop_pct=0.0, per_chunk=[],
                  verdict="n/a")
    if laps[0] > 0:
        detail["drop_pct"] = (laps[0] - laps[-1]) / laps[0] * 100
        if chunk:
            detail["per_chunk"] = [float(np.mean(laps[i:i + chunk]))
                                   for i in range(0, len(laps), chunk)]
        drop = detail["drop_pct"]
        detail["verdict"] = ("DETAIL MELT" if drop > 80 else
                             "softens" if drop > 40 else "detail holds")
    boundary = _boundary_stats(mads, chunk, overlap)

    # has_global_motion：warp/mad<0.85 且存在非零众数位移（与 video_mvp/motion_probe.py 同口径）
    mean_mad = float(np.mean(mads)) if mads else 0.0
    mean_warp = float(np.mean(warps)) if warps else 0.0
    nonzero = [(dx, dy) for dx, dy in shifts if (dx, dy) != (0, 0)]
    if nonzero:
        from collections import Counter
        best = Counter(nonzero).most_common(1)[0][0]
    else:
        best = (0, 0)
    res_ratio = (mean_warp / mean_mad) if mean_mad > 1e-9 else 1.0
    has_motion = bool(res_ratio < 0.85 and (abs(best[0]) + abs(best[1])) > 0)

    stack = np.stack(fr) if n else np.zeros((0, 0, 0, 3), dtype=np.float32)
    return dict(
        name=name,
        frames=dict(count=int(n),
                    size=[int(fr[0].shape[1]), int(fr[0].shape[0])] if n else None,
                    all_finite=bool(np.isfinite(stack).all()) if n else False,
                    std=float(stack.std()) if n else 0.0,
                    mean=float(stack.mean()) if n else 0.0,
                    frame_diff=float(np.abs(np.diff(stack, axis=0)).mean()) if n > 1 else 0.0),
        temporal=dict(mad=mean_mad, mad_first=mads[0] if mads else None,
                      warp_err=mean_warp,
                      residual=mean_mad - mean_warp,
                      hist_drift=float(np.mean(hists)) if hists else None,
                      sat_mean=float(np.mean(sats)) if sats else None,
                      best_shift=dict(dx=int(best[0]), dy=int(best[1])),
                      warp_over_mad=float(res_ratio),
                      has_global_motion=has_motion,
                      per_step_mad=mads),
        white=white, detail=detail, boundary=boundary, paths=list(paths))


def collect_groups(args):
    groups = {}
    for a in args:
        if os.path.isdir(a):
            for p in sorted(glob.glob(os.path.join(a, "*.png"))):
                groups.setdefault(os.path.basename(os.path.normpath(a)), []).append(p)
        else:
            for p in sorted(glob.glob(a)):
                base = os.path.basename(p)
                import re
                key = re.sub(r"_f\d+\.png$", "", base)
                groups.setdefault(key, []).append(p)
    out = {}
    for k, v in groups.items():
        if len(v) >= 2:
            import re
            if any(re.search(r"_f\d+\.png$", p) for p in v):
                v = sorted(v, key=lambda p: int(re.search(r"_f(\d+)\.png$", p).group(1)))
            else:
                v = sorted(v)
            out[k] = v
        else:
            print(f"(skip {k}: only {len(v)} frame(s))")
    return out


def main():
    _reconfigure_stdout()
    import argparse
    ap = argparse.ArgumentParser(description="temporal consistency metrics")
    ap.add_argument("paths", nargs="*", help="帧目录或 glob（可多个；与旧 CLI 兼容）")
    ap.add_argument("--chunk", type=int, default=None, help="每 chunk 帧数（给则算边界拆分）")
    ap.add_argument("--overlap", type=int, default=0, help="滑窗重叠帧（独立分块=0）")
    ap.add_argument("--rng", type=int, default=6, help="平移搜索半径")
    ap.add_argument("--json-out", default="", help="写统一 schema 的 metrics.json")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--extra-json", default="", help="合并辅助指标（chunk_metrics.json）")
    a = ap.parse_args()

    if not a.paths:
        print(__doc__)
        return
    groups = collect_groups(a.paths)
    if not groups:
        print("[err] 没有可分析的帧序列")
        return 2
    name, paths = next(iter(groups.items()))
    if len(groups) > 1:
        print(f"[warn] 有 {len(groups)} 组序列；json 模式只输出第一组")

    if not a.json_out:
        for k, v in groups.items():
            analyze(v, k, chunk=a.chunk, overlap=a.overlap)
        return 0

    res = analyze_sequence(paths, name, chunk=a.chunk, overlap=a.overlap, rng=a.rng)
    payload = dict(schema=SCHEMA, run_id=a.run_id or name,
                   config=dict(chunk=a.chunk, overlap=a.overlap, rng=a.rng), **res)
    payload.pop("paths", None)
    payload.pop("name", None)
    payload.setdefault("timing", {})
    payload["verdict"] = {"pass": None, "reasons": [], "warns": []}
    if a.extra_json and os.path.exists(a.extra_json):
        try:
            payload["chunks"] = json.load(open(a.extra_json, encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 合并 {a.extra_json} 失败: {e}")
    os.makedirs(os.path.dirname(os.path.abspath(a.json_out)) or ".", exist_ok=True)
    with open(a.json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[save] {a.json_out}  (schema={SCHEMA})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
