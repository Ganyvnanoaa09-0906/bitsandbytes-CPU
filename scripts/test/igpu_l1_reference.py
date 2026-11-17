# -*- coding: utf-8 -*-
"""L1 acceptance reference: a shape-generic 3-layer MLP trained on CPU (fp32).

The iGPU implementation (any device-resident fp16 path) must reproduce this loss curve
within the tolerances below. Shapes are explicit so a hard-coded M/D/FF cannot pass:

    x[M,K] -> Linear(K,H) -> SiLU -> Linear(H,H) -> SiLU -> Linear(H,N) -> y
    loss = MSE(y, target)
    20 x AdamW(lr=1e-3, betas=(0.9,0.999), eps=1e-8, wd=0)

Usage:
    python scripts/test/igpu_l1_reference.py --out ref.json
    python scripts/test/igpu_l1_reference.py --compare device.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = dict(m=512, k=1024, h=9216, n=1024, steps=20, lr=1e-3, seed=0)


def build(m, k, h, n, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(k, h), nn.SiLU(),
        nn.Linear(h, h), nn.SiLU(),
        nn.Linear(h, n),
    )
    x = torch.randn(m, k) * 0.02
    target = torch.randn(m, n) * 0.02
    return model, x, target


def tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()[:16]


def run(cfg):
    model, x, target = build(cfg["m"], cfg["k"], cfg["h"], cfg["n"], cfg["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0)
    losses, grad_norms, step_times = [], {}, []
    for step in range(cfg["steps"]):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        if step == 0:
            grad_norms = {"l1": float(model[0].weight.grad.norm()),
                          "l2": float(model[2].weight.grad.norm()),
                          "l3": float(model[4].weight.grad.norm())}
        opt.step()
        losses.append(float(loss.detach()))
        step_times.append(time.perf_counter() - t0)
    weights = {"l1": tensor_sha(model[0].weight), "l2": tensor_sha(model[2].weight),
               "l3": tensor_sha(model[4].weight)}
    return {
        "config": dict(cfg),
        "dtype": "fp32-cpu",
        "losses": losses,
        "final_loss": losses[-1],
        "first_grad_norms": grad_norms,
        "weights_sha256_16": weights,
        "step_seconds_median": statistics.median(step_times),
        "params": sum(p.numel() for p in model.parameters()),
    }


def compare(ref, dev, loss_rtol, final_rtol):
    ok = True
    if int(dev.get("steps", len(dev.get("losses", [])))) != len(ref["losses"]):
        print("  FAIL: step count mismatch: ref %d vs dev %s"
              % (len(ref["losses"]), dev.get("steps", len(dev.get("losses", [])))))
        return False
    if len(dev.get("losses", [])) != len(ref["losses"]):
        print("  FAIL: loss count mismatch")
        return False
    worst = 0.0
    for i, (a, b) in enumerate(zip(ref["losses"], dev["losses"])):
        if not (b == b) or b in (float("inf"), float("-inf")):
            print("  FAIL: non-finite loss at step %d" % i)
            return False
        rel = abs(a - b) / max(abs(a), 1e-12)
        worst = max(worst, rel)
        if rel > loss_rtol:
            print("  FAIL: step %d loss %.6g vs ref %.6g (rel %.3f > %.3f)"
                  % (i, b, a, rel, loss_rtol))
            ok = False
    frel = abs(dev["final_loss"] - ref["final_loss"]) / max(abs(ref["final_loss"]), 1e-12)
    print("  worst per-step rel diff: %.3f | final rel diff: %.3f" % (worst, frel))
    if frel > final_rtol:
        print("  FAIL: final loss outside %.3f" % final_rtol)
        ok = False
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=DEFAULTS["m"])
    ap.add_argument("--k", type=int, default=DEFAULTS["k"])
    ap.add_argument("--h", type=int, default=DEFAULTS["h"])
    ap.add_argument("--n", type=int, default=DEFAULTS["n"])
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("--loss-rtol", type=float, default=0.05)
    ap.add_argument("--final-rtol", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    cfg = dict(m=a.m, k=a.k, h=a.h, n=a.n, steps=a.steps, lr=a.lr, seed=a.seed)
    if a.compare:
        ref = json.load(open(a.out, encoding="utf-8")) if a.out else run(cfg)
        dev = json.load(open(a.compare, encoding="utf-8"))
        sys.exit(0 if compare(ref, dev, a.loss_rtol, a.final_rtol) else 1)
    result = run(cfg)
    print("params %.1fM | losses %.6g -> %.6g | step %.1f ms | grad norms %s"
          % (result["params"] / 1e6, result["losses"][0], result["final_loss"],
             result["step_seconds_median"] * 1e3, result["first_grad_norms"]))
    if a.out:
        json.dump(result, open(a.out, "w", encoding="utf-8"), indent=2)
        print("wrote", a.out)
    sys.exit(0)


if __name__ == "__main__":
    main()
