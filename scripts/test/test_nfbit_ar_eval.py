"""AR 30M checkpoint: fp32 vs NF2/NF3 vs uniform 2/3-bit eval loss + storage.

Paired statistics: every configuration is evaluated on the same samples and in the same
order, so the per-sample CE differences d_i = CE_quant(i) - CE_fp32(i) are paired. We
report mean(d), SE(d) = sigma(d)/sqrt(n) and z = mean(d)/SE(d); this is far more sensitive
than comparing two independent means (the per-sample CE sigma is ~0.95 on weibo 32x32,
so an unpaired n=512 run has SE ~0.042 and cannot resolve 0.01-level effects).
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "bitsandbytes"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
sys.stdout.reconfigure(encoding="utf-8")

from loop_transformer import LoopConfig
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2

from bitsandbytes import nfbit

CKPT = r"D:\work\cloud_salvage\oWAZY8\ckpt_3000.pt"
TOKENS = r"D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt"
N_VAL = 512
BLOCK = 64
QUICK = False
SCOPE = "all"


def evaluate(model, val):
    """Return (mean CE, per-sample CE array). All tokens except the first are scored."""
    model.eval()
    per_sample = []
    with torch.no_grad():
        for i in range(0, len(val), 8):
            x = val[i : i + 8]
            out = model.forward_logits(x, None)
            logits = out[1]
            logits = logits[0] if isinstance(logits, tuple) else logits
            token_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, model.cfg.vocab_size),
                x[:, 1:].reshape(-1),
                reduction="none",
            )
            per_sample.extend(token_loss.view(x.shape[0], -1).mean(dim=1).tolist())
    arr = np.asarray(per_sample, dtype=np.float64)
    return float(arr.mean()), arr


def paired(diff):
    """mean, standard error, z for an array of paired differences."""
    n = len(diff)
    if n < 2:
        return float(diff.mean()), float("nan"), float("nan")
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(n))
    return mean, se, (mean / se if se else float("nan"))


def should_quantize(key):
    if SCOPE == "blocks":
        return key.startswith("blocks.")
    return True


def quantize_state(sd, bits, table, block):
    q = {}
    for k, v in sd.items():
        if (
            should_quantize(k)
            and torch.is_tensor(v)
            and v.dtype == torch.float32
            and v.ndim >= 2
        ):
            packed, absmax = nfbit.quantize(v, bits=bits, block=block, table=table)
            q[k] = nfbit.dequantize(
                packed, absmax, v.shape, bits=bits, block=block, table=table
            )
        elif torch.is_tensor(v):
            q[k] = v
    return q


def storage_bytes(sd, bits, table, block):
    total = 0
    for k, v in sd.items():
        if (
            should_quantize(k)
            and torch.is_tensor(v)
            and v.dtype == torch.float32
            and v.ndim >= 2
        ):
            packed, absmax = nfbit.quantize(v, bits=bits, block=block, table=table)
            total += nfbit.storage_bytes(packed, absmax)
        elif torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return total


def main():
    global CKPT, TOKENS, N_VAL, BLOCK, QUICK, SCOPE
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--tokens", default=TOKENS)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--skip-blocks", action="store_true")
    ap.add_argument("--scope", choices=("all", "blocks"), default="all")
    args = ap.parse_args()
    CKPT, TOKENS, N_VAL, BLOCK, QUICK, SCOPE = (
        args.ckpt,
        args.tokens,
        args.n_val,
        args.block,
        args.skip_blocks,
        args.scope,
    )
    torch.set_num_threads(6)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if torch.is_tensor(v)}
    names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
    cfg = SmallImageConfigV2(
        **{k: v for k, v in (ck.get("cfg") or {}).items() if k in names}
    )
    val = torch.load(TOKENS, map_location="cpu", weights_only=False)[-N_VAL:].long()

    def build(state):
        m = SmallARImageModelV2(cfg)
        m.set_loop(
            LoopConfig(
                loop_start=cfg.loop_start,
                loop_end=cfg.loop_end,
                loop_L=1,
                loop_L_max=3,
                loop_random=False,
                loop_cache="shared",
            )
        )
        m.load_state_dict(state, strict=False)
        return m

    fp32_bytes = sum(v.numel() * v.element_size() for v in sd.values())
    t0 = time.perf_counter()
    base, base_samples = evaluate(build(sd), val)
    t_base = time.perf_counter() - t0
    print(
        f"模型 {os.path.basename(CKPT)} | 参数 {sum(v.numel() for v in sd.values()) / 1e6:.1f}M | fp32 {fp32_bytes / 2**20:.1f} MB"
    )
    print(
        "验证集 %d 条 | block %d | scope %s | fp32 loss %.4f (%.1f s) | per-sample σ=%.3f"
        % (N_VAL, BLOCK, SCOPE, base, t_base, base_samples.std(ddof=1))
    )
    print()
    print(
        "%-16s %9s %9s %9s %8s %8s %7s"
        % ("配置", "val loss", "Δ vs fp32", "SE(d)", "z", "bit/权重", "存储MB")
    )
    print("-" * 78)

    samples_by = {"fp32": base_samples}
    rows = [("fp32", None, None)]
    configs = (
        ((2, "NF"), (2, "uniform"), (3, "NF"))
        if QUICK
        else ((2, "NF"), (3, "NF"), (2, "uniform"), (3, "uniform"))
    )
    for bits, name in configs:
        rows.append(
            ("%d-bit %s" % (bits, name), bits, "nf" if name == "NF" else "uniform")
        )

    for label, bits, table in rows:
        if bits is None:
            st, loss, samples, bpp, sbytes = sd, base, base_samples, 32.0, fp32_bytes
        else:
            st = quantize_state(sd, bits, table, BLOCK)
            t0 = time.perf_counter()
            loss, samples = evaluate(build(st), val)
            dt = time.perf_counter() - t0
            sbytes = storage_bytes(sd, bits, table, BLOCK)
            bpp = sbytes * 8.0 / sum(v.numel() for v in sd.values())
            print(f"  ({label} eval {dt:.1f} s)")
        samples_by[label] = samples
        d = samples - base_samples
        mean_d, se_d, z = paired(d)
        print(
            "%-16s %9.4f %+9.4f %9.4f %8.2f %8.2f %7.1f"
            % (label, loss, mean_d, se_d, z, bpp, sbytes / 2**20)
        )
    print()
    print("配对差（同一批样本；d_i = CE_config(i) − CE_fp32(i)）：")
    for label in samples_by:
        if label == "fp32":
            continue
        d = samples_by[label] - base_samples
        mean_d, se_d, z = paired(d)
        print(
            "  %-16s mean %+.4f  SE %.4f  z %+.2f  (n=%d)"
            % (label, mean_d, se_d, z, len(d))
        )
    if "2-bit NF" in samples_by and "2-bit uniform" in samples_by:
        d = samples_by["2-bit NF"] - samples_by["2-bit uniform"]
        mean_d, se_d, z = paired(d)
        print(
            "  NF2 − uniform2   mean %+.4f  SE %.4f  z %+.2f  (n=%d)"
            % (mean_d, se_d, z, len(d))
        )
    if QUICK:
        return
    print()
    print("NF2 block 敏感度（配对差 vs fp32；大 block 理论上应更差，负值先当噪声）：")
    for block in (32, 64, 128, 256):
        st = quantize_state(sd, 2, "nf", block)
        loss, samples = evaluate(build(st), val)
        sbytes = storage_bytes(sd, 2, "nf", block)
        mean_d, se_d, z = paired(samples - base_samples)
        print(
            "  block %3d: loss %.4f  meanΔ %+.4f  SE %.4f  z %+.2f  %.2f bit/权重  %.1f MB"
            % (
                block,
                loss,
                mean_d,
                se_d,
                z,
                sbytes * 8.0 / sum(v.numel() for v in sd.values()),
                sbytes / 2**20,
            )
        )


if __name__ == "__main__":
    main()
