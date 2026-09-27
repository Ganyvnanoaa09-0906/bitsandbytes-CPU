"""用修正后的判定逻辑重算 motion_sweep 的已保存数据。

为什么要重算：扫描运行期间我改掉了判定逻辑 —— 旧版拿"合成控制样本校准出的
S2>=0.80"当"主体完整"阈值，那等于**把量具的性质当成被测物的性质**：
合成样本（平移/撕裂）不代表真实角色动画的结构变化量，按那个阈值连 k=1.0
（肉眼为"面部有错乱但人物可辨"）都会被判成崩坏。
修正后不硬塞外推阈值，改为：① 报剂量-反应曲线；② 崩溃需**多条信号同时成立**；
③ 判定以肉眼复核为准。数据本身没变，只重算判读。
"""
import json
import os
import sys

import numpy as np

SRC = r"D:\work\cloud_results\motion_sweep.json"
OUT = r"D:\work\cloud_results\motion_sweep_analyzed.json"

d = json.load(open(SRC, encoding="utf-8"))
scales = d["scales"]
smax = max(s["S1_sharpness"] for s in scales)

print("=== 剂量-反应曲线（修正判读）===")
print(f"{'k':>5s} {'实际比值':>9s} {'M1(px)':>8s} {'方向一致':>9s} {'闪烁残差':>9s} "
      f"{'S1':>7s} {'S1/峰值':>8s} {'S2min':>7s} {'S2中位':>7s} {'S3抖动':>7s} {'判定':>9s}")
rows = []
for s in scales:
    r = s["S1_sharpness"] / smax
    sig = {
        "方向反转(一致性<0)": s["direction_consistency"] < 0.0,
        "闪烁爆炸(>=60)": s["flicker"] >= 60.0,
        "锐度丢失(S1/峰值<0.35)": r < 0.35,
    }
    n = sum(1 for v in sig.values() if v)
    verdict = "COLLAPSE" if n >= 2 else ("DEGRADED" if n == 1 else "OK")
    rows.append({**s, "S1_vs_best": r, "signals": sig, "collapse_count": n,
                 "verdict": verdict})
    print(f"{s['k']:5.1f} {s['applied_ratio']:9.3f} {s['M1_motion_px']:8.2f} "
          f"{s['direction_consistency']:9.2f} {s['flicker']:9.2f} "
          f"{s['S1_sharpness']:7.0f} {r:8.3f} {(s['S2_structure_min'] or 0):7.3f} "
          f"{(s['S2_structure_median'] or 0):7.3f} {s['S3_jitter']:7.1f} {verdict:>9s}")

print("\n=== 逐条信号 ===")
for r in rows:
    hits = [k for k, v in r["signals"].items() if v]
    print(f"  k={r['k']}: {r['collapse_count']} 条 —— {hits if hits else '无'}")

# ---- 关键对比：k=1.0 vs k>=1.3 的跃变 ----
base = next(r for r in rows if r["k"] == 1.0)
aggr = [r for r in rows if r["k"] >= 1.3]
print("\n=== 跃变（k=1.0 → k>=1.3）===")
if aggr:
    m1 = [r["M1_motion_px"] for r in aggr]
    print(f"  位移       {base['M1_motion_px']:.2f}px  →  {min(m1):.2f}~{max(m1):.2f}px"
          f"  （×{min(m1)/max(base['M1_motion_px'],1e-9):.1f} ~ ×{max(m1)/max(base['M1_motion_px'],1e-9):.1f}）")
    print(f"  方向一致性 {base['direction_consistency']:+.2f}  →  "
          f"{min(r['direction_consistency'] for r in aggr):+.2f}~{max(r['direction_consistency'] for r in aggr):+.2f}")
    print(f"  锐度       {base['S1_sharpness']:.0f}  →  "
          f"{min(r['S1_sharpness'] for r in aggr):.0f}~{max(r['S1_sharpness'] for r in aggr):.0f}")
    print(f"  结构相关   {(base['S2_structure_min'] or 0):.3f}  →  "
          f"{min((r['S2_structure_min'] or 0) for r in aggr):.3f}~{max((r['S2_structure_min'] or 0) for r in aggr):.3f}")

# ---- 判定 ----
non_collapse = [r for r in rows if r["verdict"] != "COLLAPSE"]
best = max(non_collapse, key=lambda r: r["M1_motion_px"]) if non_collapse else None
checks = [
    ("扫描覆盖从非崩坏到崩坏（判据有区分度）",
     len(non_collapse) > 0 and len(non_collapse) < len(rows)),
    ("运动与崩坏同向：位移最大的那组崩得最厉害",
     max(rows, key=lambda r: r["M1_motion_px"])["verdict"] == "COLLAPSE"),
    ("存在位移>=1px 且未崩坏的配置",
     bool(best) and best["M1_motion_px"] >= 1.0),
    ("k=1.0 是唯一方向连贯的点（|一致性|最大）",
     base["direction_consistency"] == max(abs(r["direction_consistency"]) for r in rows)),
]
print("\n=== 判定 ===")
npass = 0
for n, o in checks:
    print(f"  [{'PASS' if o else 'FAIL'}] {n}")
    npass += bool(o)
print(f"\n{npass}/{len(checks)} PASS")
if best:
    print(f"⇒ 非崩坏组里位移最大：k={best['k']}，M1={best['M1_motion_px']:.2f}px，"
          f"方向={best['direction_consistency']:+.2f}，S1/峰值={best['S1_vs_best']:.3f}")
else:
    print("⇒ 全部崩坏（阴性结果）")

json.dump({"rows": rows, "checks": [{"name": n, "pass": bool(o)} for n, o in checks],
           "passed": npass, "total": len(checks),
           "best_non_collapse_k": best["k"] if best else None},
          open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
print(f"\n[已写入] {OUT}")
sys.exit(0)
