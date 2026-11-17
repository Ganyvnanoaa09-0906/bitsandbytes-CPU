# -*- coding: utf-8 -*-
"""igpu_l3_device.py — L3 device-side runner for LEAD's CPU reference.

Usage:
    python gt1030/igpu_l3_device.py \
        --ref D:\\work\\bitsandbytes-CPU\\scripts\\test\\igpu_l3_reference.py \
        --out device_l3.json

The reference file is imported for `DEFAULTS`/`build_model`/`make_data`, so both sides
start from exactly the same weights and data (per its docstring).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gt1030.igpu_block as B  # noqa: E402


def load_ref(path):
    spec = importlib.util.spec_from_file_location("igpu_l3_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref", default=r"D:\work\bitsandbytes-CPU\scripts\test\igpu_l3_reference.py"
    )
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--layers-per-ctx", type=int, default=1)
    args = ap.parse_args()
    ref = load_ref(args.ref)
    cfg = dict(ref.DEFAULTS)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.n_layers is not None:
        cfg["n_layers"] = args.n_layers

    model = ref.build_model(cfg, cfg["seed"])
    x, target = ref.make_data(cfg, cfg["seed"])
    rows, d, ffn = cfg["b"] * cfg["t"], cfg["d"], cfg["ffn"]
    x_np = x.reshape(rows, d).numpy().astype(np.float16)
    t_np = target.reshape(rows, d).numpy().astype(np.float32)

    st = B.Stack(
        rows,
        d,
        ffn,
        cfg["nh"],
        cfg["hd"],
        cfg["t"],
        cfg["n_layers"],
        seed=cfg["seed"],
        eps=cfg["eps"],
        final_norm=False,
        per_layer=True,
    )
    with torch.no_grad():
        for rb, db in zip(model.blocks, st.blocks):
            db.activate()
            db.norm1.set_weight(rb.n1.w.detach().numpy())
            db.norm2.set_weight(rb.n2.w.detach().numpy())
            db.attn.qkv.set_weight(rb.attn.qkv.weight.detach().numpy())
            db.attn.wo.set_weight(rb.attn.out.weight.detach().numpy())
            w12 = rb.mlp.w12.weight.detach().numpy()
            db.mlp.gate.set_weight(w12[:ffn])
            db.mlp.up.set_weight(w12[ffn:])
            db.mlp.down.set_weight(rb.mlp.w3.weight.detach().numpy())

    losses, step_times = [], []
    for step in range(1, cfg["steps"] + 1):
        t0 = time.perf_counter()
        yh = st.forward_np(x_np)
        loss = float(((yh - t_np) ** 2).mean())
        # fp16 MSE per-element dy is ~1e-6 (< fp16 min normal); scale by 65536 and
        # scale AdamW eps by the same factor.
        G = 65536.0
        dy = (2.0 * (yh - t_np) / (rows * d) * G).astype(np.float16)
        st.backward_np(dy)
        st.step(cfg["lr"], step, eps=1e-8 * 65536.0)
        step_times.append(time.perf_counter() - t0)
        losses.append(loss)
        if step in (1, cfg["steps"]):
            print("  step %2d: %.6g" % (step, loss))

    res = {
        "config": cfg,
        "dtype": "fp16-igpu",
        "backend": "CLBlast/OpenCL gfx902",
        "losses": losses,
        "final_loss": losses[-1],
        "step_seconds_median": statistics.median(step_times),
        "params": sum(p.numel() for p in model.parameters()),
    }
    print(
        "  params %.2fM | losses %.6g -> %.6g | step %.1f ms"
        % (res["params"] / 1e6, losses[0], losses[-1], res["step_seconds_median"] * 1e3)
    )
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
        print("  wrote", args.out)


if __name__ == "__main__":
    main()
