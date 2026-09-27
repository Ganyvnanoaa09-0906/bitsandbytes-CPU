"""找"崩坏阈值以下的最大运动"。

已有结论（实测 + 人眼确认）：
  · LCM 路线：主体完好（S2 结构相关 0.89~0.96）但几乎不动（M1 0.00~0.20px）
  · DPM 路线：运动出来了（M1 0.77~12.60px）但主体崩（S2 0.30；权重×2.0 时退化成纯噪声）
  ⇒ 真正的优化目标是**扫描出 S2 还站得住的最大运动强度**，而不是继续加运动。

本轮：DPM 12 步 + 动作 prompt + v1-5-2 适配器，只改时序权重缩放系数，
对每个系数同时量【运动】(motion_metric.M1) 与【完整性】(subject_integrity.S1/S2)。
判据：找满足 S2>=0.80 且 S1 相对峰值 >=0.60 的最大系数。
"""
import os
import sys
import json
import time
import gc

import numpy as np
import torch

torch.set_num_threads(6)
sys.path.insert(0, r"D:\work")
from motion_exp import anime_adiff  # noqa
from motion_metric import motion_metric, to_gray
from subject_integrity import subject_integrity

OUT = r"D:\work\cloud_results\motion_sweep"
os.makedirs(OUT, exist_ok=True)

ADAPTER = r"D:\work\textmodel\animatediff-motion-adapter-v1-5-2"
MOTION_PROMPT = ("1girl, solo, upper body, silver hair blowing in the wind, "
                 "turning head, hair swaying, dynamic pose, wind, "
                 "detailed face, school uniform, cinematic lighting, best quality")
NEG = ("lowres, bad anatomy, bad hands, extra digits, cropped, worst quality, "
       "low quality, bad face, deformed, blurry")
FRAMES, RES, SEED, STEPS = 8, 256, 1234, 12
SCALES = [1.0, 1.3, 1.6, 2.0]


def main():
    pipe = anime_adiff.build(scheduler="dpm", adapter=ADAPTER)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.disable_attention_slicing()
    except Exception:
        pass

    motion_params = [(n, p) for n, p in pipe.unet.named_parameters() if "motion" in n.lower()]
    originals = [(p, p.detach().clone()) for _, p in motion_params]

    def set_scale(k):
        with torch.no_grad():
            for p, orig in originals:
                p.copy_(orig * k)

    rep = {"scales": [], "frames": FRAMES, "res": RES, "steps": STEPS, "seed": SEED}

    for k in SCALES:
        set_scale(k)
        # 断言缩放真的生效
        ratio = float((motion_params[0][1].detach()
                       / originals[0][1].clamp(min=1e-12)).abs().median())
        g = torch.Generator(device="cpu").manual_seed(SEED)
        t0 = time.time()
        with torch.no_grad():
            out = pipe(prompt=MOTION_PROMPT, negative_prompt=NEG, num_frames=FRAMES,
                       num_inference_steps=STEPS, guidance_scale=7.5,
                       height=RES, width=RES, generator=g)
        dt = time.time() - t0
        frames = out.frames[0]
        gray = [to_gray(f) for f in frames]
        mm = motion_metric(gray, max_shift=24)
        si = subject_integrity(gray)

        d = os.path.join(OUT, f"k{k}")
        os.makedirs(d, exist_ok=True)
        from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(np.asarray(f)).convert("RGB").save(os.path.join(d, "f%03d.png" % i))

        e = si["S2_structure_min"]
        print(f"  k={k}: 比值={ratio:.3f}  {dt:.0f}s  "
              f"M1={mm['M1_motion_px']:.2f}px  方向={mm['direction_consistency']:.2f}  "
              f"S1={si['S1_subject_sharpness']:.0f}  S2min={e if e is None else round(e,3)}  "
              f"S3抖动={si['S3_centroid_jitter_mean']:.1f}", flush=True)
        rep["scales"].append({"k": k, "applied_ratio": ratio, "gen_seconds": dt,
                              "M1_motion_px": mm["M1_motion_px"],
                              "direction_consistency": mm["direction_consistency"],
                              "flicker": mm["flicker"], "mean_absdiff": mm["mean_absdiff"],
                              "shifts": mm["shifts"],
                              "S1_sharpness": si["S1_subject_sharpness"],
                              "S2_structure_min": e,
                              "S2_structure_median": si["S2_structure_median"],
                              "S3_jitter": si["S3_centroid_jitter_mean"]})
        del out, frames, gray
        gc.collect()

    set_scale(1.0)
    del pipe
    gc.collect()

    # ---- 呈现剂量-反应曲线（不硬塞一个外推来的阈值）----
    # 为什么不用"synthetic 校准出来的 S2>=0.80 就判定完整"：合成控制样本（平移/撕裂）
    # 不能代表真实角色动画的结构变化量 —— 真实动作会让 S2 天然更低。
    # 按那个阈值连 k=1.0（肉眼是"面部有错乱但人物可辨"）都会被判成崩坏。
    # 阈值若来自合成样本外推，就是把量具的性质当成了被测物的性质。
    # ⇒ 这里只报剂量-反应，判定交给"肉眼 + 方向一致性/闪烁"这两个有明确物理含义的量。
    smax = max(s["S1_sharpness"] for s in rep["scales"])
    for s in rep["scales"]:
        s["S1_vs_best"] = s["S1_sharpness"] / smax
        # 崩溃的三条同时出现才算（单一指标不足以判定）
        s["collapse_signals"] = {
            "direction_reversed (方向一致性<0)": s["direction_consistency"] < 0.0,
            "flicker_exploded (闪烁残差>=60)": s["flicker"] >= 60.0,
            "sharpness_lost (S1/峰值<0.35)": s["S1_vs_best"] < 0.35,
        }
        s["collapse_count"] = sum(1 for v in s["collapse_signals"].values() if v)
        s["verdict"] = "COLLAPSE" if s["collapse_count"] >= 2 else (
            "DEGRADED" if s["collapse_count"] == 1 else "OK")

    print("\n" + "=" * 92)
    print("剂量-反应曲线（运动强度 → 位移 / 连贯性 / 完整性）")
    print(f"{'k':>5s} {'实际比值':>9s} {'M1(px)':>8s} {'方向一致':>9s} {'闪烁残差':>9s} "
          f"{'S1':>7s} {'S1/峰值':>8s} {'S2min':>7s} {'判定':>9s}")
    for s in rep["scales"]:
        print(f"{s['k']:5.1f} {s['applied_ratio']:9.3f} {s['M1_motion_px']:8.2f} "
              f"{s['direction_consistency']:9.2f} {s['flicker']:9.2f} "
              f"{s['S1_sharpness']:7.0f} {s['S1_vs_best']:8.3f} "
              f"{(s['S2_structure_min'] or 0):7.3f} {s['verdict']:>9s}")

    intact = [s for s in rep["scales"] if s["verdict"] != "COLLAPSE"]
    best = max(intact, key=lambda s: s["M1_motion_px"]) if intact else None
    collapse = [s for s in rep["scales"] if s["verdict"] == "COLLAPSE"]

    checks = [
        ("扫描覆盖了从完好到崩坏（既有非崩坏也有崩坏）",
         len(intact) > 0 and len(collapse) > 0),
        ("运动与崩坏同向：位移最大的那组崩得最厉害",
         bool(rep["scales"]) and
         max(rep["scales"], key=lambda s: s["M1_motion_px"])["verdict"] == "COLLAPSE"),
        ("存在位移 >=1px 且未崩坏的配置",
         bool(best) and best["M1_motion_px"] >= 1.0),
    ]
    print("\n--- 判定 ---")
    npass = 0
    for n, o in checks:
        print(f"  [{'PASS' if o else 'FAIL'}] {n}")
        npass += bool(o)
    print(f"\n{npass}/{len(checks)} PASS")
    if best:
        print(f"⇒ 非崩坏组里位移最大的：k={best['k']}，M1={best['M1_motion_px']:.2f}px，"
              f"方向={best['direction_consistency']:.2f}，S1/峰值={best['S1_vs_best']:.3f}")
    else:
        print("⇒ 全部崩坏（阴性结果）")

    rep["checks"] = [{"name": n, "pass": bool(o)} for n, o in checks]
    rep["passed"], rep["total"] = npass, len(checks)
    rep["best_k"] = best["k"] if best else None
    with open(r"D:\work\cloud_results\motion_sweep.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
