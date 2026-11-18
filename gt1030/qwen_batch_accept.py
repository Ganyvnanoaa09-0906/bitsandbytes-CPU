# -*- coding: utf-8 -*-
"""qwen_batch_accept.py — 多题材验收：用 qwen_img.py 的 preset 批量出图，记录每张的耗时与成败。

用法：python qwen_batch_accept.py --preset 1024 --outdir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

PROMPTS = [
    ("animal_fox", "a red fox standing in a snowy pine forest at golden hour, wildlife photo, sharp detail"),
    ("portrait_fisher", "portrait of an elderly fisherman with a weathered face and grey beard, natural window light, 85mm photo, shallow depth of field"),
    ("landscape_lake", "a misty mountain lake at sunrise, pine trees along the shore, mirror reflection, landscape photo"),
    ("object_teapot", "a celadon ceramic teapot on a wooden table, soft studio light, product photo, clean background"),
    ("text_shop", "a small coffee shop storefront with a chalkboard sign reading \"COFFEE\", warm morning light, street photo"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="1024")
    ap.add_argument("--outdir", default=r"D:\work\research\bench_results\qwen_batch")
    ap.add_argument("--igpu", default="none")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    py = sys.executable
    cli = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_img.py")
    rows = []
    for tag, prompt in PROMPTS:
        out = os.path.join(a.outdir, f"{tag}_{a.preset}.png")
        if os.path.exists(out):
            print(f"  [skip] {tag} 已存在", flush=True)
            rows.append(dict(tag=tag, out=out, skipped=True))
            continue
        t0 = time.time()
        cmd = [py, "-u", cli, "--prompt", prompt, "--preset", a.preset, "--out", out]
        if a.igpu != "none":
            cmd += ["--igpu", a.igpu]
        print(f"\n=== [{tag}] {prompt[:60]}...", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.time() - t0
        tail = [l for l in (p.stdout or "").splitlines() if "RESULT" in l or "save" in l or "Traceback" in l]
        ok = p.returncode == 0 and os.path.exists(out)
        print(f"   rc={p.returncode} {dt:.0f}s ok={ok}", flush=True)
        for l in tail[-3:]:
            print("   " + l[:200], flush=True)
        if not ok:
            print("   stderr tail:", (p.stderr or "")[-400:].replace("\n", " | "), flush=True)
        rows.append(dict(tag=tag, prompt=prompt, out=out, seconds=round(dt, 1), ok=ok))
    with open(os.path.join(a.outdir, "batch_report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n=== 汇总 ===", flush=True)
    for r in rows:
        print("  %-18s %s" % (r["tag"], ("skip" if r.get("skipped") else
                                         ("OK %.0fs" % r["seconds"] if r.get("ok") else "FAIL"))), flush=True)


if __name__ == "__main__":
    main()
