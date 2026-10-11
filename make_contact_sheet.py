# -*- coding: utf-8 -*-
"""Build a contact sheet of the motion-boundary sweep so the whole boundary is visible at once.

Picks frame 5 (mid-clip, past any first-frame settling) from each level and lays them out in
a labelled grid:

    row 1:  L1 micro | L2 headturn | L3 wave | L4 runjump      (escalating motion)
    row 2:  L5 fight | MA closeup  | MB bokeh | (blank)

Each tile is annotated with its key metrics (det% / ncc / radial streak alignment) so the
numbers and the picture can be read together.

Usage:
  python make_contact_sheet.py --out D:\work\output\motion\contact_sheet.png
"""
import argparse
import json
import os

from PIL import Image, ImageDraw

MOTION = r"D:\work\output\motion"
MIT = r"D:\work\output\motion_mit"
CELL = 512
PAD = 8
LABEL_H = 74


def load_metrics():
    m = {}
    for root in (MOTION, MIT):
        for name in ("motion.json", "face.json"):
            p = os.path.join(root, name)
            if os.path.exists(p):
                try:
                    d = json.load(open(p, encoding="utf-8"))
                except Exception:
                    continue
                for k, v in d.items():
                    m.setdefault(k, {}).update(v.get("summary", v) if isinstance(v, dict) else {})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tiles = [
        ("L1 micro", os.path.join(MOTION, "L1_micro", "frames", f"f{a.frame:03d}.png")),
        ("L2 headturn", os.path.join(MOTION, "L2_headturn", "frames", f"f{a.frame:03d}.png")),
        ("L3 wave", os.path.join(MOTION, "L3_wave", "frames", f"f{a.frame:03d}.png")),
        ("L4 runjump", os.path.join(MOTION, "L4_runjump", "frames", f"f{a.frame:03d}.png")),
        ("L5 fight", os.path.join(MOTION, "L5_fight", "frames", f"f{a.frame:03d}.png")),
        ("MA closeup+bokeh", os.path.join(MIT, "MA_closeup", "frames", f"f{a.frame:03d}.png")),
        ("MB blur+bokeh", os.path.join(MIT, "MB_bokeh", "frames", f"f{a.frame:03d}.png")),
    ]
    met = load_metrics()

    cols, rows = 4, 2
    W = cols * CELL + (cols + 1) * PAD
    H = rows * (CELL + LABEL_H) + (rows + 1) * PAD
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    dr = ImageDraw.Draw(sheet)

    keys = ["L1_micro", "L2_headturn", "L3_wave", "L4_runjump", "L5_fight",
            "MA_closeup", "MB_bokeh"]
    for i, (label, path) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = PAD + c * (CELL + PAD)
        y = PAD + r * (CELL + LABEL_H + PAD)
        if not os.path.exists(path):
            dr.text((x + 8, y + 8), f"{label}: MISSING", fill=(255, 120, 120))
            continue
        sheet.paste(Image.open(path).convert("RGB"), (x, y))
        k = keys[i] if i < len(keys) else label
        m = met.get(k, {})
        det = m.get("det_rate")
        ncc = m.get("ncc")
        ra = m.get("radial_align")
        sub = []
        if det is not None:
            sub.append(f"det {100*det:.0f}%")
        if ncc is not None:
            sub.append(f"ncc {ncc:.3f}")
        if ra is not None:
            sub.append(f"radial {ra:.2f}")
        # flag the collapse boundary visually. NOTE: det% is meaningless for extreme
        # close-ups -- keypointrcnn cannot detect a "person" in a face-only crop -- so the
        # close-up tile is excluded from the automatic flag.
        flag = ""
        if det is not None and det < 0.95 and not label.startswith("MA"):
            flag = "  <-- DEGRADED"
        dr.text((x + 6, y + CELL + 6), label + flag, fill=(255, 255, 255))
        if label.startswith("MA"):
            dr.text((x + 6, y + CELL + 26), "face-only crop; det% not applicable",
                    fill=(160, 220, 255))
        else:
            dr.text((x + 6, y + CELL + 26), "  ".join(sub), fill=(160, 220, 255))
        if ra is not None and ra < 0.637:
            dr.text((x + 6, y + CELL + 46), "radial streaks = effect masking",
                    fill=(255, 200, 120))

    sheet.save(a.out)
    print(f"[save] {a.out}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
