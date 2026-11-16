#!/usr/bin/env python3
"""Turn numerics.json into a compact markdown table for the report / archive."""

import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "numerics.json"
rows = json.load(open(path, encoding="utf-8"))

by_bs = defaultdict(list)
for r in rows:
    by_bs[r["blocksize"]].append(r)

print("### weight-only quantization error (relative Frobenius), fp32-absmax vs ue8m0\n")
print("| distribution | BK | absmax | ue8m0 | ue8m0/absmax |")
print("|---|---|---|---|---|")
for bs in sorted(by_bs):
    for r in by_bs[bs]:
        ratio = r["weight_err_ue8m0"] / r["weight_err_absmax"]
        print(f"| {r['dist']} | {bs} | {r['weight_err_absmax']:.3e} | {r['weight_err_ue8m0']:.3e} | {ratio:.2f}x |")

print("\n### end-to-end GEMM error (y = x @ W^T, x ~ t(5))\n")
print("| distribution | BK | float-act + absmax-W | float-act + ue8m0-W | int8-act + ue8m0-W |")
print("|---|---|---|---|---|")
for bs in sorted(by_bs):
    for r in by_bs[bs]:
        print(
            f"| {r['dist']} | {bs} | {r['gemm_err_absmax']:.3e} | "
            f"{r['gemm_err_ue8m0']:.3e} | {r['gemm_err_int8_both']:.3e} |"
        )

# Aggregate ratios
import statistics

w_ratio = statistics.mean(r["weight_err_ue8m0"] / r["weight_err_absmax"] for r in rows)
g_ratio = statistics.mean(r["gemm_err_ue8m0"] / r["gemm_err_absmax"] for r in rows)
i8_ratio = statistics.mean(r["gemm_err_int8_both"] / r["gemm_err_absmax"] for r in rows)
print("\n### summary ratios (mean over all 25 cases)\n")
print(f"- ue8m0 weight error / absmax weight error        : {w_ratio:.2f}x")
print(f"- ue8m0 GEMM error   / absmax GEMM error          : {g_ratio:.2f}x")
print(f"- int8-both GEMM err / float-act absmax GEMM err  : {i8_ratio:.2f}x")
