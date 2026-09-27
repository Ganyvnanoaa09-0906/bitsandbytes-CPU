"""先确认"画的是什么"，再谈运动 —— 别对抽象输出做运动测量。

报告 §10.7/§10.8 记录过：本项目的生成输出曾是"抽象色场"（token 坍塌）。
如果这次 motion_exp 的输出也是抽象色块，那对它做块位移匹配得到的"0.00 px"
是在测量无意义内容，不能当成"模型不动"的证据 —— 那属于把量具的产物当成被测物的属性。

逐组输出：亮度分布、颜色数、水平/垂直边缘能量、以及"是否存在一个可辨认主体"
（前景/背景的对比度）。同时把每组的第 0 帧并排拼一张对比图，供人眼确认。
"""
import os
import glob
import json

import numpy as np
from PIL import Image

ROOT = r"D:\work\cloud_results\motion_exp"
OUT_JSON = r"D:\work\cloud_results\motion_exp_content.json"
OUT_SHEET = os.path.join(ROOT, "_contact_sheet.png")


def stats(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, dtype=np.float32)
    g = a.mean(axis=2)
    # 颜色丰富度：唯一颜色的粗略量化（每通道 5 级）
    q = (a // 51).astype(np.uint8)
    uniq = len(np.unique(q.reshape(-1, 3), axis=0))
    # 边缘能量：水平/垂直梯度均值（有细节才高）
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    # 前景/背景对比：亮度直方图的动态范围
    p5, p95 = np.percentile(g, 5), np.percentile(g, 95)
    return {
        "size": list(im.size),
        "mean": float(g.mean()),
        "std": float(g.std()),
        "p5": float(p5), "p95": float(p95),
        "dynamic_range": float(p95 - p5),
        "unique_quant_colors": int(uniq),
        "edge_x": float(gx), "edge_y": float(gy),
        "edge_mean": float((gx + gy) / 2),
    }


def main():
    if not os.path.isdir(ROOT):
        print("无输出目录")
        return 1
    groups = sorted(d for d in os.listdir(ROOT)
                    if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith("_"))
    if not groups:
        print("还没有任何组完成")
        return 1

    rep = {}
    sheets = []
    for g in groups:
        fs = sorted(glob.glob(os.path.join(ROOT, g, "*.png")))
        if not fs:
            continue
        per_frame = [stats(f) for f in fs]
        agg = {k: float(np.mean([p[k] for p in per_frame]))
               for k in ("mean", "std", "dynamic_range", "unique_quant_colors",
                         "edge_x", "edge_y", "edge_mean")}
        agg["n_frames"] = len(fs)
        rep[g] = agg
        print(f"\n--- {g}  ({len(fs)} 帧) ---")
        for k, v in agg.items():
            print(f"    {k:22s} {v:.2f}" if isinstance(v, float) else f"    {k:22s} {v}")
        # 供人眼看的第 0 帧
        sheets.append((g, fs[0]))

    if sheets:
        ims = [Image.open(p).convert("RGB") for _, p in sheets]
        w, h = ims[0].size
        sheet = Image.new("RGB", (w * len(ims), h), (0, 0, 0))
        for i, im in enumerate(ims):
            sheet.paste(im.resize((w, h)), (i * w, 0))
        sheet.save(OUT_SHEET)
        print(f"\n[对比图] {OUT_SHEET}  顺序: {[g for g, _ in sheets]}")

    # 判定：输出是不是"有内容的画面"
    checks = []
    for g, a in rep.items():
        checks.append((f"{g}: 有可辨内容 (std>15 且 边缘>3)",
                       a["std"] > 15 and a["edge_mean"] > 3))
    print("\n--- 判定（内容是否有意义）---")
    npass = 0
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        npass += bool(ok)
    print(f"\n{npass}/{len(checks)} 组是有效画面")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"groups": rep,
                   "checks": [{"name": n, "pass": bool(o)} for n, o in checks]},
                  f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
