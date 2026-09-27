"""运动判据：区分【真实运动】与【逐帧闪烁】。

为什么需要它
------------
现有 `anime_video_e2e.py` 的"动态"判据是 `mean|帧间差分| > 0.1`。上一轮输出
帧间差分 11.32 看着很大，但报告已查明那**全部来自超分幻觉闪烁**，主体几乎没动。
也就是说：旧判据把闪烁判成了运动。

两种信号的物理区别
------------------
· 真实运动：相邻帧之间的差异是一块**有关联**的区域在**有方向地**位移 ⇒
  最优块位移 (dx,dy) 显著非零、幅值大，且**方向在时间上连续**。
· 逐帧闪烁：每帧的差异是**独立随机**的细小变化 ⇒ 最优块位移接近 0，
  剩下的能量是"无法被任何位移解释"的残差。

判据（两个都要过）
------------------
  M1  全局运动幅值  |motion| = 平均最优块位移长度（像素）  >= 阈值
  M2  运动可解释度  explain = 1 - 残差能量/总差异能量     >= 阈值
      （闪烁的 explain 接近 0：位移补偿不了它）

控制样本（先用合成数据验证判据本身有效，再拿它评真实输出）
  · 静态：同一张图重复 8 次        -> 期望 M1≈0
  · 平移：一张图每帧平移 3 px      -> 期望 M1≈3、explain 高
  · 闪烁：同一张图 + 独立随机噪声   -> 期望 M1≈0、explain 低
"""
import os
import json
import numpy as np


# ----------------------------------------------------------------------
# 块匹配：对每对相邻帧求"最能解释差异"的位移
# ----------------------------------------------------------------------
def _best_shift(a, b, max_shift=16, block=48, stride=12):
    """把 b 相对 a 平移 (dx,dy)，找使残差最小的位移。返回 (dx,dy,res_energy,var_energy)。"""
    h, w = a.shape
    # 只在能完整取块的区域比较
    m = max_shift
    if h <= 2 * m + block or w <= 2 * m + block:
        return 0.0, 0.0, 0.0, float(np.abs(b - a).mean())

    var_energy = float(np.abs(b[m:h - m, m:w - m] - a[m:h - m, m:w - m]).mean())
    best = None
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            bs = b[m + dy:h - m + dy, m + dx:w - m + dx]
            r = float(np.abs(bs - a[m:h - m, m:w - m]).mean())
            if best is None or r < best[2]:
                best = (dx, dy, r)
    return best[0], best[1], best[2], var_energy


def motion_metric(frames, max_shift=16, verbose=False):
    """frames: list of 2D float arrays (灰度)。返回运动统计。"""
    a_frames = [np.asarray(f, dtype=np.float32) for f in frames]
    if len(a_frames) < 2:
        return {"n": len(a_frames), "M1_motion_px": 0.0, "M2_explain": 0.0,
                "flicker": 0.0, "shifts": []}

    shifts, explains, flicks, residuals = [], [], [], []
    static_pairs = 0
    for i in range(len(a_frames) - 1):
        a, b = a_frames[i], a_frames[i + 1]
        dx, dy, res, var = _best_shift(a, b, max_shift=max_shift)
        mag = float(np.hypot(dx, dy))
        # var=0 表示两帧在比较区内完全相同 ⇒ "无差异"，可解释度是 0/0 无定义，
        # 不能塞一个数进去（塞 0 会把"完全静止"报成"完全不可解释"）。
        if var > 1e-9:
            exp = 1.0 - (res / var)
        else:
            exp = None
            static_pairs += 1
        shifts.append((int(dx), int(dy), mag))
        if exp is not None:
            explains.append(exp)
        flicks.append(res)
        residuals.append(res)

    mags = [s[2] for s in shifts]
    return {
        "n": len(a_frames),
        "M1_motion_px": float(np.mean(mags)),
        "M1_motion_max": float(np.max(mags)),
        "M2_explain": (float(np.mean(explains)) if explains else None),
        "M2_explain_defined_pairs": len(explains),
        "static_pairs": static_pairs,
        "flicker": float(np.mean(flicks)),
        "mean_absdiff": float(np.mean([np.abs(a_frames[i + 1] - a_frames[i]).mean()
                                       for i in range(len(a_frames) - 1)])),
        "shifts": [(s[0], s[1]) for s in shifts],
        "direction_consistency": _direction_consistency(shifts),
    }


def _direction_consistency(shifts):
    """相邻位移的方向是否连续（真实运动给 0~1，随机给 ~0）。"""
    if len(shifts) < 2:
        return 0.0
    dots = []
    for i in range(len(shifts) - 1):
        u = np.array(shifts[i][:2], dtype=float)
        v = np.array(shifts[i + 1][:2], dtype=float)
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-6 or nv < 1e-6:
            continue
        dots.append(float(np.dot(u, v) / (nu * nv)))
    return float(np.mean(dots)) if dots else 0.0


def to_gray(img):
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        return a[..., :3].mean(axis=2)
    return a


# ----------------------------------------------------------------------
# 合成控制样本：先证明判据本身能用
# ----------------------------------------------------------------------
def selfcheck():
    rng = np.random.default_rng(0)
    h = w = 192
    # 造一张有结构的图（不是纯噪声，块匹配才有意义）
    yy, xx = np.mgrid[0:h, 0:w]
    base = (60.0 + 80.0 * np.sin(xx / 14.0) * np.cos(yy / 11.0)
            + 50.0 * (((xx - 96) ** 2 + (yy - 96) ** 2) < 900))

    def shifted(n, px):
        return [np.roll(np.roll(base, int(round(px * i)), axis=1), 0, axis=0)
                for i in range(n)]

    def static(n):
        return [base.copy() for _ in range(n)]

    def flicker(n, amp=30.0):
        return [base + rng.normal(0, amp, base.shape) for _ in range(n)]

    cases = {
        "static": static(8),
        "pan_3px": shifted(8, 3.0),
        "pan_6px": shifted(8, 6.0),
        "flicker": flicker(8, 30.0),
        "flicker_strong": flicker(8, 60.0),
    }
    out = {}
    print("=== 判据自检（合成控制样本）===")
    print(f"{'样本':16s} {'M1运动(px)':>11s} {'M2可解释':>10s} {'闪烁残差':>10s} {'平均差分':>10s} {'方向一致':>9s}")
    for name, fr in cases.items():
        m = motion_metric(fr, max_shift=12)
        out[name] = m
        e = m["M2_explain"]
        es = "   无差异" if e is None else f"{e:10.3f}"
        print(f"{name:16s} {m['M1_motion_px']:11.2f} {es} "
              f"{m['flicker']:10.2f} {m['mean_absdiff']:10.2f} {m['direction_consistency']:9.3f}")

    # 判定：判据必须能分开这三类，否则它没有用
    checks = [
        ("静态 M1≈0", out["static"]["M1_motion_px"] < 0.6),
        ("静态可解释度无定义（不假装有值）", out["static"]["M2_explain"] is None),
        ("静态所有帧对都无差异", out["static"]["static_pairs"] == 7),
        ("平移3px M1≈3", abs(out["pan_3px"]["M1_motion_px"] - 3.0) < 1.0),
        ("平移6px M1≈6", abs(out["pan_6px"]["M1_motion_px"] - 6.0) < 1.2),
        ("平移可解释度≈1", out["pan_3px"]["M2_explain"] > 0.9),
        ("闪烁可解释度低", out["flicker"]["M2_explain"] < 0.35),
        ("强闪烁可解释度更低", out["flicker_strong"]["M2_explain"] < out["flicker"]["M2_explain"] + 0.05),
        ("平移方向一致>0.9", out["pan_3px"]["direction_consistency"] > 0.9),
        ("闪烁运动幅值≈0", out["flicker"]["M1_motion_px"] < 1.0),
        ("判据能分开平移与闪烁", (out["pan_3px"]["M1_motion_px"] > 2.0
                                  and out["flicker"]["M1_motion_px"] < 1.0)),
    ]
    print("\n--- 判定 ---")
    npass = 0
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        npass += bool(ok)
    print(f"\n{npass}/{len(checks)} PASS")
    os.makedirs(r"D:\work\cloud_results", exist_ok=True)
    with open(r"D:\work\cloud_results\motion_metric_selfcheck.json", "w", encoding="utf-8") as f:
        json.dump({"cases": {k: v for k, v in out.items()},
                   "checks": [{"name": n, "pass": bool(o)} for n, o in checks],
                   "passed": npass, "total": len(checks)}, f, indent=2, ensure_ascii=False)
    return npass == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if selfcheck() else 1)
