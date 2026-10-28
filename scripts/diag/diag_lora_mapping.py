# -*- coding: utf-8 -*-
"""Diagnose the LoRA merge failure: is the path mapping between the trained 2D UNet and the
inference-time UNetMotionModel actually valid?

Three checks, cheapest first:

 1. weight identity -- for each of the 128 LoRA target paths, compare the tensor in
    UNet2DConditionModel (what training wrapped) against the same path in UNetMotionModel
    (what inference merges into). If they differ, the mapping is invalid and that alone
    explains garbage output.
 2. module type -- confirm each target is an nn.Linear whose weight shape equals B@A.
 3. merge arithmetic -- report |B@A| / |W| per layer class, to see whether the delta is
    genuinely tiny (which would make total output destruction impossible and point at a bug
    in the merge code rather than in the numbers).

Usage:
  python diag_lora_mapping.py --lora D:\work\lora_char2_run500\final
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file as lf

BASE = r"D:\work\textmodel\counterfeit-v3-diffusers"
ADAPTER = r"D:\work\textmodel\animatediff-motion-adapter-v1-5"


def collect(sd):
    pairs = {}
    for k, v in sd.items():
        if not k.startswith("base_model.model."):
            continue
        rest = k[len("base_model.model."):]
        if rest.endswith(".lora_A.weight"):
            pairs.setdefault(rest[:-len(".lora_A.weight")], {})["A"] = v
        elif rest.endswith(".lora_B.weight"):
            pairs.setdefault(rest[:-len(".lora_B.weight")], {})["B"] = v
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=r"D:\work\lora_char2_run500\final")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from diffusers import UNet2DConditionModel, MotionAdapter
    from diffusers.models import UNetMotionModel

    print("[load] 2D UNet (training target) ...", flush=True)
    u2d = UNet2DConditionModel.from_pretrained(os.path.join(BASE, "unet"),
                                               torch_dtype=torch.float32)
    print("[load] UNetMotionModel (inference target) ...", flush=True)
    ma = MotionAdapter.from_pretrained(ADAPTER, torch_dtype=torch.float32)
    um = UNetMotionModel.from_unet2d(u2d, ma)

    m2d = dict(u2d.named_modules())
    mum = dict(um.named_modules())

    sd = lf(os.path.join(a.lora, "adapter_model.safetensors"))
    pairs = collect(sd)
    print(f"\n[keys] {len(sd)} tensors -> {len(pairs)} LoRA target layers\n")

    rows = []
    identical = missing_m = missing_2d = shape_bad = type_bad = 0
    ratios = []
    for path, ab in sorted(pairs.items()):
        r = dict(path=path)
        if path not in m2d:
            missing_2d += 1
            r["status"] = "absent in 2D UNet"
            rows.append(r)
            continue
        if path not in mum:
            missing_m += 1
            r["status"] = "absent in MotionModel"
            rows.append(r)
            continue
        w2 = getattr(m2d[path], "weight", None)
        wm = getattr(mum[path], "weight", None)
        if w2 is None or wm is None:
            r["status"] = "no weight attr"
            rows.append(r)
            continue
        same = bool(torch.equal(w2, wm))
        ba = ab["B"].float() @ ab["A"].float()
        shape_ok = tuple(ba.shape) == tuple(wm.shape)
        type_ok = type(mum[path]).__name__ == "Linear"
        if same:
            identical += 1
        if not shape_ok:
            shape_bad += 1
        if not type_ok:
            type_bad += 1
        ratio = float(ba.norm() / wm.norm())
        ratios.append(ratio)
        r.update(status="ok" if same else "WEIGHT MISMATCH",
                 same=same, type=type(mum[path]).__name__, shape_ok=shape_ok,
                 ratio=ratio, w_norm=float(wm.norm()), ba_norm=float(ba.norm()))
        rows.append(r)

    print(f"{'check':<34} {'result':>10}")
    print("-" * 46)
    print(f"{'layers found in both models':<34} {identical + (len(rows)-identical-missing_2d-missing_m):>10}")
    print(f"{'weight IDENTICAL (2D vs Motion)':<34} {identical:>10}")
    print(f"{'weight MISMATCH':<34} {len(rows)-identical-missing_2d-missing_m:>10}")
    print(f"{'absent in 2D UNet':<34} {missing_2d:>10}")
    print(f"{'absent in MotionModel':<34} {missing_m:>10}")
    print(f"{'non-Linear target':<34} {type_bad:>10}")
    print(f"{'B@A shape mismatch':<34} {shape_bad:>10}")
    if ratios:
        print(f"\n|B@A|/|W|: mean={np.mean(ratios):.4f} max={np.max(ratios):.4f} "
              f"min={np.min(ratios):.4f}")

    bad = [r for r in rows if r.get("status") == "WEIGHT MISMATCH"]
    if bad:
        print(f"\n*** {len(bad)} WEIGHT MISMATCHES -- the merge targets the WRONG tensors ***")
        print("examples:")
        for r in bad[:8]:
            print(f"   {r['path'][:70]:<70} |W|={r['w_norm']:.3f} ratio={r['ratio']:.4f}")
    else:
        print("\nAll mapped weights are identical -> mapping is valid; look elsewhere "
              "for the merge bug.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dict(rows=rows, identical=identical, mismatch=len(bad),
                           missing_2d=missing_2d, missing_motion=missing_m,
                           shape_bad=shape_bad, type_bad=type_bad,
                           ratios=ratios), f, indent=2)
        print(f"[save] {a.json}")


if __name__ == "__main__":
    main()
