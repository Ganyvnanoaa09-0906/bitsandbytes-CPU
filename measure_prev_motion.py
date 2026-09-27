"""用新运动判据复测【上一轮的既有输出】—— 建立基线。

报告 §10.151 附近已记：上一轮 8 帧输出的"帧间差分 11.32"主要来自超分幻觉闪烁。
现在用能区分"运动 / 闪烁"的判据把它量出来，看这个说法是否成立。

对两套帧都测：
  lr_frames/  生成器原始输出（256px）—— 闪烁应当是**超分引入的**，所以这里更干净
  sr_frames/  超分后（1024px）—— 报告说闪烁主要在这里
"""
import os
import sys
import glob
import json

import numpy as np
from PIL import Image

sys.path.insert(0, r"D:\work")
from motion_metric import motion_metric, to_gray

ROOT = r"D:\work\cloud_results\anime_e2e"
OUT = r"D:\work\cloud_results\motion_baseline_prev.json"


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "*.png")))
    return [to_gray(Image.open(f).convert("RGB")) for f in fs], fs


def main():
    rep = {}
    for tag, d in (("lr_256px", os.path.join(ROOT, "lr_frames")),
                   ("sr_1024px", os.path.join(ROOT, "sr_frames"))):
        if not os.path.isdir(d):
            print(f"  {tag}: 目录不存在")
            continue
        frames, files = load(d)
        if len(frames) < 2:
            print(f"  {tag}: 只有 {len(frames)} 帧，跳过")
            continue
        # 两次：一次用生成分辨率的位移范围，一次用大图的范围
        m = motion_metric(frames, max_shift=16)
        rep[tag] = {"dir": d, "n_frames": len(frames), **m}
        print(f"\n=== {tag}  ({len(frames)} 帧, {frames[0].shape}) ===")
        print(f"  M1 平均运动幅值      {m['M1_motion_px']:.2f} px   (最大 {m['M1_motion_max']:.2f})")
        e = m["M2_explain"]
        print(f"  M2 运动可解释度      {'无差异' if e is None else f'{e:.3f}'}"
              f"   (有效帧对 {m['M2_explain_defined_pairs']}/{len(frames)-1})")
        print(f"  闪烁残差（无法解释）  {m['flicker']:.2f}")
        print(f"  平均帧间差分          {m['mean_absdiff']:.2f}   ← 旧判据用的就是这个")
        print(f"  位移序列              {m['shifts']}")
        print(f"  方向一致性            {m['direction_consistency']:.3f}")

    # 判定：报告的说法（运动极小、差分来自闪烁）是否成立
    lr = rep.get("lr_256px")
    sr = rep.get("sr_1024px")
    checks = []
    if lr:
        checks.append(("LR 真实运动幅值很小 (<2px)", lr["M1_motion_px"] < 2.0))
    if sr:
        checks.append(("SR 真实运动幅值很小 (<2px)", sr["M1_motion_px"] < 2.0))
    if lr and sr:
        checks.append(("SR 的帧间差分 > LR（超分放大了差异）",
                       sr["mean_absdiff"] > lr["mean_absdiff"]))
        checks.append(("两套都表现为闪烁而非运动",
                       (lr["M2_explain"] is None or lr["M2_explain"] < 0.5)
                       and (sr["M2_explain"] is None or sr["M2_explain"] < 0.5)))
    print("\n--- 判定（报告说法是否成立）---")
    npass = 0
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        npass += bool(ok)
    rep["checks"] = [{"name": n, "pass": bool(o)} for n, o in checks]
    rep["passed"], rep["total"] = npass, len(checks)
    print(f"\n{npass}/{len(checks)} PASS")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
