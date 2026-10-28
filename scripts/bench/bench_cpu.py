# -*- coding: utf-8 -*-
"""bench_cpu.py — CPU training memory/bandwidth/concurrency benchmark (baseline).

Measures the three things that decide whether pure-CPU training scales:
  A. achievable memory bandwidth (copy / scale / add) in GB/s -> roofline for GEMM;
  B. dense fp32 GEMM throughput (GFLOPS) on LLM-ish shapes;
  C. concurrency: N independent training jobs at once -> aggregate throughput and
     memory per job; finds the knee where more jobs no longer help.

Usage:
    python bench_cpu.py                 # full run
    python bench_cpu.py --quick         # smaller sizes / fewer concurrency points
    python bench_cpu.py --json out.json # also save results

Worker mode (used internally for C):
    python bench_cpu.py --worker --steps 20 --threads 2 --hidden 512 --batch 4 --seq 64
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

MB = 1 << 20

# ---------------------------------------------------------------- worker


def run_worker(args) -> int:
    """One training job: synthetic decoder-ish MLP + 8-bit AdamW, fixed step count."""
    import torch
    import torch.nn as nn
    import psutil

    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "bitsandbytes"))
    import bitsandbytes as bnb

    class Block(nn.Module):
        def __init__(self, h, inter):
            super().__init__()
            self.w1 = nn.Linear(h, inter, bias=False)
            self.w2 = nn.Linear(inter, h, bias=False)

        def forward(self, x):
            return x + self.w2(torch.nn.functional.silu(self.w1(x)))

    h = args.hidden
    model = nn.Sequential(Block(h, 4 * h), Block(h, 4 * h), Block(h, 4 * h))
    # freeze half the params: representative of LoRA/EFST-style training (cold params)
    for i, p in enumerate(model.parameters()):
        p.requires_grad_(i % 2 == 0)
    x = torch.randn(args.batch, args.seq, h)
    target = torch.randn(args.batch, args.seq, h)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(params, lr=1e-4)

    step_times = []
    for step in range(args.steps):
        t0 = time.perf_counter()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step_times.append(time.perf_counter() - t0)

    rss = psutil.Process().memory_info().rss / MB
    med = statistics.median(step_times[2:] if len(step_times) > 3 else step_times)
    tokens_per_step = args.batch * args.seq
    print(json.dumps({
        "threads": args.threads,
        "steps": args.steps,
        "step_ms_median": med * 1e3,
        "steps_per_s": 1.0 / med if med else 0.0,
        "tokens_per_s": tokens_per_step / med if med else 0.0,
        "rss_mb": rss,
        "loss_final": float(loss.detach()),
    }))
    return 0


# ---------------------------------------------------------------- memory bandwidth


def stream_bandwidth(quick: bool = False):
    """Read/copy/scale/add bandwidth in GB/s (2N bytes counted for copy-like ops)."""
    import torch

    n = (64 if quick else 128) * 1024 * 1024 // 4  # floats
    a = torch.randn(n)
    b = torch.empty_like(a)
    res = {}
    for name, fn, nbytes in (
        ("copy(read+write)", lambda: b.copy_(a), 2 * n * 4),
        ("scale_inplace(r+w)", lambda: a.mul_(1.0000001), 2 * n * 4),
        ("add(r+r+w)", lambda: torch.add(a, a, out=b), 3 * n * 4),
    ):
        fn()  # warmup
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        dt = (time.perf_counter() - t0) / reps
        res[name] = {"GB/s": nbytes / dt / 1e9, "bytes": nbytes}
    return res


# ---------------------------------------------------------------- GEMM


def gemm_gflops(quick: bool = False):
    import torch

    torch.set_num_threads(os.cpu_count() or 1)
    shapes = [
        (512, 1024, 3584),   # LLM ffn up (batch*seq=512)
        (512, 3584, 1024),   # LLM ffn down
        (512, 1024, 1024),   # attention proj
        (2048, 2048, 2048),  # square reference
    ]
    if quick:
        shapes = shapes[:2] + [(1024, 1024, 1024)]
    out = {}
    for m, k, n in shapes:
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        torch.matmul(a, b)
        reps = 3
        t0 = time.perf_counter()
        for _ in range(reps):
            c = torch.matmul(a, b)
        dt = (time.perf_counter() - t0) / reps
        gflops = 2 * m * k * n / dt / 1e9
        out[f"{m}x{k}x{n}"] = {"GFLOPS": gflops, "ms": dt * 1e3}
        del a, b, c
    return out


# ---------------------------------------------------------------- concurrency


def concurrency(points, steps: int, threads: int, hidden: int, batch: int, seq: int):
    """Run N jobs in parallel, measure aggregate throughput and per-job RSS."""
    worker = os.path.abspath(__file__)
    results = []
    for n in points:
        procs = []
        for _ in range(n):
            cmd = [sys.executable, worker, "--worker", "--steps", str(steps),
                   "--threads", str(threads), "--hidden", str(hidden),
                   "--batch", str(batch), "--seq", str(seq)]
            procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True))
        rows = []
        t0 = time.perf_counter()
        for p in procs:
            out, _ = p.communicate()
            line = (out or "").strip().splitlines()[-1] if out else ""
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"error": line[:200]})
        wall = time.perf_counter() - t0
        good = [r for r in rows if "step_ms_median" in r]
        agg = sum(r["steps_per_s"] for r in good)
        results.append({
            "jobs": n,
            "ok_jobs": len(good),
            "wall_s": wall,
            "aggregate_steps_per_s": agg,
            "aggregate_tokens_per_s": sum(r["tokens_per_s"] for r in good),
            "mean_step_ms": statistics.mean(r["step_ms_median"] for r in good) if good else None,
            "rss_per_job_mb": statistics.mean(r["rss_mb"] for r in good) if good else None,
        })
    # efficiency vs single-job baseline
    base = results[0]["aggregate_steps_per_s"] if results and results[0]["jobs"] == 1 else None
    for r in results:
        r["scale_efficiency"] = (r["aggregate_steps_per_s"] / (base * r["jobs"])
                                 if base and r["jobs"] else None)
    return results


def packing_experiment(steps: int, hidden: int, batch: int, seq: int):
    """Which packing wins: one job with all cores, or N jobs with cores/N threads each?

    This is the practical answer for "support concurrency": a CPU with C cores cannot run
    N jobs x C threads each -- the scheduler just context-switches. Measure the plausible
    packings and report aggregate tokens/s.
    """
    cores = os.cpu_count() or 4
    configs = []
    for jobs in (1, 2, 3, cores):
        threads = max(1, cores // jobs)
        configs.append((jobs, threads))
    # also a 1-thread-each over-subscription point to show the cliff
    configs.append((cores + 2, 1))

    results = []
    for jobs, threads in configs:
        row = concurrency([jobs], steps, threads, hidden, batch, seq)[0]
        row["threads_per_job"] = threads
        row["total_threads"] = jobs * threads
        results.append(row)
    best = max(results, key=lambda r: r["aggregate_steps_per_s"])
    for r in results:
        r["pct_of_best"] = 100.0 * r["aggregate_steps_per_s"] / best["aggregate_steps_per_s"]
        r["oversubscribed"] = r["total_threads"] > cores
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--packing", action="store_true",
                    help="only run the thread-packing experiment")
    args = ap.parse_args()

    if args.worker:
        return run_worker(args)

    import torch
    print("=" * 78)
    print("CPU training benchmark | torch %s | %d logical cores | %s"
          % (torch.__version__, os.cpu_count() or 0, sys.platform))
    print("=" * 78)

    print("\nA. memory bandwidth (GB/s)")
    bw = stream_bandwidth(args.quick)
    for name, v in bw.items():
        print("  %-20s %8.2f GB/s" % (name, v["GB/s"]))

    print("\nB. dense fp32 GEMM (threads=%d)" % (os.cpu_count() or 1))
    gm = gemm_gflops(args.quick)
    for name, v in gm.items():
        print("  %-16s %8.1f GFLOPS  (%.2f ms)" % (name, v["GFLOPS"], v["ms"]))

    print("\nC. concurrency: independent training jobs (threads/job=%d)" % args.threads)
    points = [1, 2] if args.quick else [1, 2, 4, 6]
    conc = concurrency(points, 10 if args.quick else 20, args.threads,
                       args.hidden, args.batch, args.seq)
    for r in conc:
        print("  jobs=%d ok=%d  agg=%6.2f steps/s (%8.0f tok/s)  step=%6.1f ms  "
              "rss/job=%5.0f MB  eff=%s"
              % (r["jobs"], r["ok_jobs"], r["aggregate_steps_per_s"],
                 r["aggregate_tokens_per_s"], r["mean_step_ms"] or 0,
                 r["rss_per_job_mb"] or 0,
                 ("%.0f%%" % (100 * r["scale_efficiency"]))
                 if r["scale_efficiency"] is not None else "-"))

    print("\nD. packing: jobs x threads/job (same total work per job)")
    pack = packing_experiment(10 if args.quick else 15, args.hidden, args.batch, args.seq)
    for r in pack:
        print("  jobs=%-2d threads/job=%-2d total=%-2d  agg=%6.2f steps/s (%8.0f tok/s)  "
              "step=%6.1f ms  best=%5.1f%%%s"
              % (r["jobs"], r["threads_per_job"], r["total_threads"],
                 r["aggregate_steps_per_s"], r["aggregate_tokens_per_s"],
                 r["mean_step_ms"] or 0, r["pct_of_best"],
                 "  OVERSUBSCRIBED" if r["oversubscribed"] else ""))
    best_pack = max(pack, key=lambda r: r["aggregate_steps_per_s"])
    print("  -> best: jobs=%d, threads/job=%d, %.0f tok/s aggregate"
          % (best_pack["jobs"], best_pack["threads_per_job"],
             best_pack["aggregate_tokens_per_s"]))

    result = {"bandwidth": bw, "gemm": gm, "concurrency": conc, "packing": pack}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("\njson ->", args.json)
    print("\nBENCH DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
