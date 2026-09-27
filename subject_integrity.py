"""主体完整性判据：量"人在不在、有没有崩"，而不是"整幅画面有没有平移"。

为什么换判据
------------
我之前用的 `motion_metric` 量的是**全局最优块位移**。它适合"镜头平移 / 整体位移"，
但对**角色动画**是错的：主体可以一边大幅做动作、一边待在原地，全局位移接近 0。
实测印证了这一点 —— 五组输出的 M1 里四组恰好是 0.00，而画面肉眼看是正常的动漫人物。
⇒ 那个数字只说明"没有整体平移"，不能说明"没在动"。用户的判断是对的：
出视频没问题，问题是**强动作会让人物主体崩坏**。

崩坏在像素上长什么样（这是判据的依据）
  1. 面部/主体**细节塌陷**：清晰度（Laplacian 能量）下降；
  2. **结构撕裂/重影**：相邻帧的高频结构不再对齐；
  3. **位置漂移抖动**：最锐区域的质心逐帧乱跳。

判据
  S1 subject_sharpness   最锐区域的 Laplacian 能量（主体细节还在不在）
  S2 structure_min       相邻帧"局部结构相关"的最小值（撕裂/重影 ⇒ 掉下去）
  S3 centroid_jitter     最锐区域质心的帧间位移（抖动）

合成控制（先证明判据能用，再拿它评真实输出）
  · sharp_static   同一张清晰图重复           -> S1 高、S2≈1、S3≈0
  · motion_ok      整体位移（局部结构不变）    -> S1 高、S2 高、S3>0
  · blur_mild      逐帧递增模糊               -> S1 下降
  · blur_heavy     强模糊                     -> S1 很低
  · tear           逐帧随机撕裂（半幅拼接）    -> S2 掉下去
"""
import numpy as np
from scipy import ndimage


def _lap_energy(g):
    lap = ndimage.laplace(g)
    return float((lap ** 2).mean())


def _sharpest_region(g, frac=0.35):
    """返回最锐区域（Laplacian 能量最高）的位置与能量。用分块找，鲁棒且便宜。"""
    h, w = g.shape
    bh, bw = max(8, h // 4), max(8, w // 4)
    best = (-1.0, 0, 0, 0, 0)
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            blk = g[y:y + bh, x:x + bw]
            e = _lap_energy(blk)
            if e > best[0]:
                best = (e, y, x, bh, bw)
    return best


def _centroid(g, thr_ratio=0.5):
    """高频能量质心（主体位置代理）。"""
    e = np.abs(ndimage.laplace(g))
    if e.max() <= 1e-9:
        return (g.shape[0] / 2.0, g.shape[1] / 2.0)
    m = e >= e.max() * thr_ratio
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return (g.shape[0] / 2.0, g.shape[1] / 2.0)
    return (float(ys.mean()), float(xs.mean()))


def _local_structure_corr(a, b, win=16, max_shift=12):
    """局部归一化相关：**先按最优整体位移对齐**，再比结构。

    为什么必须对齐（第一版就栽在这里）：不对齐时，整体平移 3px 会让逐窗口比较
    全面失配，S2 从 1.0 掉到 0.495 —— 于是判据把"位移"误判成"撕裂"，
    正是这个任务最不该犯的错。对齐后，位移不再影响 S2，只有真正的结构破坏
    （重影/撕裂/形变不一致）才会让它下降。
    """
    dx, dy = _best_global_shift(a, b, max_shift=max_shift)
    # ⚠️ 符号约定必须与 _best_global_shift 一致。该函数的约定是
    #   "b 在 (y,x) 的值取自 b[y-dy, x-dx]"，因此用 np.roll 实现时要取负号。
    # 第一版写成 np.roll(b, dy) 把位移**加倍**了（3px -> 6px），
    # 代价是结构相关从 1.0 掉到 0.325，于是判据把"平移"误判成"撕裂"。
    # 手工验证：roll(-3) 残差 0.0000，roll(+3) 残差 25.10。
    if dx or dy:
        b = np.roll(np.roll(b, -dy, axis=0), -dx, axis=1)
    # 对齐后裁掉位移带来的边缘回绕区，避免用假的周期内容比较
    m = abs(max_shift)
    h, w = a.shape
    if h <= 2 * m + win or w <= 2 * m + win:
        aa, bb = a, b
    else:
        aa = a[m:h - m, m:w - m]
        bb = b[m:h - m, m:w - m]
    hh, ww = aa.shape
    vals = []
    for y in range(0, hh - win + 1, win):
        for x in range(0, ww - win + 1, win):
            pa = aa[y:y + win, x:x + win].ravel()
            pb = bb[y:y + win, x:x + win].ravel()
            sa, sb = pa.std(), pb.std()
            if sa < 1e-6 or sb < 1e-6:
                continue
            c = float(np.corrcoef(pa, pb)[0, 1])
            if np.isfinite(c):
                vals.append(c)
    return float(np.median(vals)) if vals else 0.0


def _best_global_shift(a, b, max_shift=12, step=2):
    """粗到细找使 |a-b| 最小的整体位移。step=2 起步足够，S2 只需近似对齐。"""
    def cost(dx, dy):
        m = max_shift
        h, w = a.shape
        if h <= 2 * m or w <= 2 * m:
            return float("inf")
        aa = a[m:h - m, m:w - m]
        bb = b[m + dy:h - m + dy, m + dx:w - m + dx]
        return float(np.abs(aa - bb).mean())

    best = (0, 0, float("inf"))
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            c = cost(dx, dy)
            if c < best[2]:
                best = (dx, dy, c)
    # 在最优附近精修
    bx, by = best[0], best[1]
    for dy in range(by - step + 1, by + step):
        for dx in range(bx - step + 1, bx + step):
            if abs(dx) > max_shift or abs(dy) > max_shift:
                continue
            c = cost(dx, dy)
            if c < best[2]:
                best = (dx, dy, c)
    return best[0], best[1]


def subject_integrity(frames, verbose=False):
    gs = [np.asarray(f, dtype=np.float32) for f in frames]
    if gs[0].ndim == 3:
        gs = [g[..., :3].mean(axis=2) for g in gs]

    sharp = []
    cent = []
    for g in gs:
        e, y, x, bh, bw = _sharpest_region(g)
        sharp.append(e)
        cent.append(_centroid(g))

    struct = [_local_structure_corr(gs[i], gs[i + 1]) for i in range(len(gs) - 1)]
    jitter = [float(np.hypot(cent[i + 1][0] - cent[i][0], cent[i + 1][1] - cent[i][1]))
              for i in range(len(cent) - 1)]

    s1 = float(np.median(sharp))
    out = {
        "n": len(gs),
        "S1_subject_sharpness": s1,
        "S1_first": sharp[0] if sharp else 0.0,
        "S1_last": sharp[-1] if sharp else 0.0,
        "S1_ratio_last_over_first": (sharp[-1] / sharp[0]) if sharp and sharp[0] > 1e-9 else None,
        "S1_min_over_median": (min(sharp) / s1) if s1 > 1e-9 else None,
        "S2_structure_min": float(min(struct)) if struct else None,
        "S2_structure_median": float(np.median(struct)) if struct else None,
        "S3_centroid_jitter_mean": float(np.mean(jitter)) if jitter else 0.0,
        "S3_centroid_jitter_max": float(np.max(jitter)) if jitter else 0.0,
    }
    if verbose:
        print(f"    sharpness 逐帧: {[round(s, 1) for s in sharp]}")
        print(f"    structure 逐帧对: {[round(s, 3) for s in struct]}")
        print(f"    centroid jitter: {[round(j, 2) for j in jitter]}")
    return out


# ----------------------------------------------------------------------
# 合成控制样本
# ----------------------------------------------------------------------
def _base_image(h=192, w=192, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = (60 + 70 * np.sin(xx / 13.0) * np.cos(yy / 10.0)
           + 60 * (((xx - 96) ** 2 + (yy - 96) ** 2) < 1200))
    # 加一些高频细节（模拟面部/头发纹理）
    img = img + 18 * rng.normal(0, 1, (h, w))
    return img.astype(np.float32)


def selfcheck():
    base = _base_image()
    rng = np.random.default_rng(1)

    cases = {
        "sharp_static": [base.copy() for _ in range(8)],
        "motion_ok": [np.roll(base, i * 3, axis=1) for i in range(8)],
        "blur_mild": [ndimage.gaussian_filter(base, 1.0 * i) for i in range(8)],
        "blur_heavy": [ndimage.gaussian_filter(base, 3.0) for _ in range(8)],
        "tear": [np.concatenate([base[: 96], np.roll(base[96:], i * 11, axis=1)], axis=0)
                 for i in range(8)],
    }
    res = {}
    print("=== 主体完整性判据自检（合成控制）===")
    print(f"{'样本':14s} {'S1锐度':>10s} {'S1末/首':>9s} {'S2结构最小':>11s} "
          f"{'S2结构中位':>11s} {'S3抖动':>8s}")
    for name, fr in cases.items():
        m = subject_integrity(fr)
        res[name] = m
        r = m["S1_ratio_last_over_first"]
        print(f"{name:14s} {m['S1_subject_sharpness']:10.1f} "
              f"{('None' if r is None else f'{r:.3f}'):>9s} "
              f"{m['S2_structure_min']:11.3f} {m['S2_structure_median']:11.3f} "
              f"{m['S3_centroid_jitter_mean']:8.2f}")

    s = res
    checks = [
        ("清晰静态：锐度高且末/首≈1",
         s["sharp_static"]["S1_ratio_last_over_first"] > 0.95),
        ("清晰静态：结构相关≈1", s["sharp_static"]["S2_structure_min"] > 0.95),
        ("平移：结构相关仍高（位移不等于崩坏）",
         s["motion_ok"]["S2_structure_min"] > 0.7),
        ("轻度模糊：锐度下降", s["blur_mild"]["S1_ratio_last_over_first"] < 0.7),
        ("撕裂：结构相关明显下降（<0.9）",
         s["tear"]["S2_structure_min"] < 0.9),
        ("撕裂比平移的结构相关更低",
         s["tear"]["S2_structure_min"] < s["motion_ok"]["S2_structure_min"]),
        ("平移有位移抖动（S3>0）", s["motion_ok"]["S3_centroid_jitter_mean"] > 0.5),
    ]

    # ---- 模糊阶梯标定 ----
    # 说明：S1 是"最锐区域的 Laplacian 能量"，它对**同一内容**有效，跨内容不可比。
    # 所以"末/首≈1"对"每帧同样模糊"的序列报 1.0 是**正常**的，不能拿它测恒定模糊。
    # 正确做法：对同一张图做递增模糊，看 S1 如何随模糊程度单调下降 —— 这条曲线才是
    # 解释真实输出 S1 数值的标尺。
    print("\n=== 模糊阶梯标定（同一张图的 S1 对模糊程度）===")
    print(f"{'sigma':>6s} {'S1锐度':>12s} {'相对清晰图':>12s}")
    ladder = []
    ref = _lap_energy(_sharpest_region(base)[0:1] and base) if False else None
    base_s1 = s["sharp_static"]["S1_subject_sharpness"]
    for sigma in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
        img = base if sigma == 0 else ndimage.gaussian_filter(base, sigma)
        v = subject_integrity([img] * 4)["S1_subject_sharpness"]
        ladder.append({"sigma": sigma, "S1": v, "ratio_to_sharp": v / base_s1})
        print(f"{sigma:6.1f} {v:12.1f} {v / base_s1:12.4f}")

    mono = all(ladder[i]["S1"] >= ladder[i + 1]["S1"] for i in range(len(ladder) - 1))
    checks.append(("模糊阶梯单调下降（S1 是有效的清晰度标尺）", mono))
    checks.append(("sigma=3 时锐度掉到清晰图的 20% 以下",
                   ladder[4]["ratio_to_sharp"] < 0.2))
    res["_blur_ladder"] = ladder
    print("\n--- 判定 ---")
    npass = 0
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        npass += bool(ok)
    print(f"\n{npass}/{len(checks)} PASS")
    return npass == len(checks), res, checks


if __name__ == "__main__":
    import json
    import os
    import sys
    ok, res, checks = selfcheck()
    os.makedirs(r"D:\work\cloud_results", exist_ok=True)
    with open(r"D:\work\cloud_results\subject_integrity_selfcheck.json", "w",
              encoding="utf-8") as f:
        json.dump({"cases": res,
                   "checks": [{"name": n, "pass": bool(o)} for n, o in checks]},
                  f, indent=2, ensure_ascii=False, default=str)
    sys.exit(0 if ok else 1)
