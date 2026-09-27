"""用主体完整性判据评真实输出：运动加大到什么程度，人物开始崩？

横轴 = 运动强度（时序权重缩放 / 适配器版本）
纵轴 = S1 主体锐度、S2 结构相关、S3 质心抖动
判据解释见 subject_integrity.py（已用合成控制验证 9/9，平移 S2=1.000、撕裂 S2=0.852）
"""
import os
import glob
import json

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, r"D:\work")
from subject_integrity import subject_integrity

ROOTS = [r"D:\work\cloud_results\motion_fix",
         r"D:\work\cloud_results\motion_exp"]
OUT_JSON = r"D:\work\cloud_results\subject_integrity_real.json"
OUT_SHEET = r"D:\work\cloud_results\motion_breakdown_sheet.png"


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "*.png")))
    out = []
    for f in fs:
        out.append(np.asarray(Image.open(f).convert("RGB"), dtype=np.float32).mean(axis=2))
    return out, fs


def main():
    groups = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d) or name.startswith("_"):
                continue
            fr, fs = load(d)
            if len(fr) < 2:
                continue
            m = subject_integrity(fr)
            groups.append({"group": f"{os.path.basename(root)}/{name}", "dir": d,
                           "n": len(fr), **m})

    if not groups:
        print("没有可评的输出")
        return 1

    # 用清晰静态的 S1 当"未崩坏"参考：取所有组里 S1 的最大值
    smax = max(g["S1_subject_sharpness"] for g in groups)
    print(f"参考：所有组中最高的 S1 主体锐度 = {smax:.1f}\n")
    print(f"{'组':30s} {'S1锐度':>9s} {'相对峰值':>9s} {'S2结构最小':>10s} "
          f"{'S2中位':>8s} {'S3抖动':>8s} {'判定':>8s}")
    for g in groups:
        ratio = g["S1_subject_sharpness"] / smax
        s2 = g["S2_structure_min"]
        # 判定：锐度保留 >50% 且结构相关 >0.85 视为"未崩"
        ok = ratio > 0.5 and (s2 is None or s2 > 0.85)
        g["sharpness_vs_best"] = ratio
        g["verdict"] = "OK" if ok else "BROKEN"
        print(f"{g['group']:30s} {g['S1_subject_sharpness']:9.1f} {ratio:9.3f} "
              f"{(s2 if s2 is not None else float('nan')):10.3f} "
              f"{g['S2_structure_median']:8.3f} {g['S3_centroid_jitter_mean']:8.2f} "
              f"{g['verdict']:>8s}")

    # 拼一张对比图：每组第 4 帧（中间帧）并排
    try:
        ims = []
        labels = []
        for g in groups:
            fs = sorted(glob.glob(os.path.join(g["dir"], "*.png")))
            if len(fs) >= 5:
                ims.append(Image.open(fs[4]).convert("RGB"))
                labels.append(g["group"].replace("motion_fix/", "").replace("motion_exp/", ""))
        if ims:
            w, h = ims[0].size
            sheet = Image.new("RGB", (w * len(ims), h), (0, 0, 0))
            for i, im in enumerate(ims):
                sheet.paste(im.resize((w, h)), (i * w, 0))
            sheet.save(OUT_SHEET)
            print(f"\n[对比图] {OUT_SHEET}\n  顺序: {labels}")
    except Exception as e:
        print(f"拼图失败: {e}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"reference_sharpness": smax, "groups": groups},
                  f, indent=2, ensure_ascii=False, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
