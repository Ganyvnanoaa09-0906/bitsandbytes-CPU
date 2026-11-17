"""AR 30M checkpoint: fp32 vs NF2/NF3 vs uniform 2/3-bit eval loss + storage.

This is the independent acceptance check for the NF storage format: the numbers it
prints must agree with the nf3/nf2 study's independently measured
fp32 5.5038 / 2-bit NF 5.6206 / 2-bit uniform 5.6354 (same checkpoint, same block 64).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time

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


def evaluate(model, val):
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val), 8):
            x = val[i : i + 8]
            out = model.forward_logits(x, None)
            logits = out[1]
            logits = logits[0] if isinstance(logits, tuple) else logits
            tot += float(
                torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, model.cfg.vocab_size),
                    x[:, 1:].reshape(-1),
                    reduction="sum",
                )
            )
            cnt += x[:, 1:].numel()
    return tot / max(cnt, 1)


def quantize_state(sd, bits, table, block):
    q = {}
    for k, v in sd.items():
        if torch.is_tensor(v) and v.dtype == torch.float32 and v.ndim >= 2:
            packed, absmax = nfbit.quantize(v, bits=bits, block=block, table=table)
            q[k] = nfbit.dequantize(
                packed, absmax, v.shape, bits=bits, block=block, table=table
            )
        elif torch.is_tensor(v):
            q[k] = v
    return q


def storage_bytes(sd, bits, table, block):
    total = 0
    for v in sd.values():
        if torch.is_tensor(v) and v.dtype == torch.float32 and v.ndim >= 2:
            packed, absmax = nfbit.quantize(v, bits=bits, block=block, table=table)
            total += nfbit.storage_bytes(packed, absmax)
        elif torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return total


def main():
    global CKPT, TOKENS, N_VAL, BLOCK, QUICK
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--tokens", default=TOKENS)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--skip-blocks", action="store_true")
    args = ap.parse_args()
    CKPT, TOKENS, N_VAL, BLOCK, QUICK = args.ckpt, args.tokens, args.n_val, args.block, args.skip_blocks
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
    base = evaluate(build(sd), val)
    t_base = time.perf_counter() - t0
    print(
        f"模型 {os.path.basename(CKPT)} | 参数 {sum(v.numel() for v in sd.values()) / 1e6:.1f}M | fp32 {fp32_bytes / 2**20:.1f} MB"
    )
    print(
        "验证集 %d 条 | block %d | fp32 loss %.4f (%.1f s)"
        % (N_VAL, BLOCK, base, t_base)
    )
    print()
    print(
        "%-18s %10s %10s %10s %10s %8s"
        % ("配置", "val loss", "Δ vs fp32", "bit/权重", "存储 MB", "压缩比")
    )
    print("-" * 74)

    rows = [
        ("fp32", None, None),
    ]
    configs = ((2, "NF"), (2, "uniform"), (3, "NF")) if QUICK else (
        (2, "NF"), (3, "NF"), (2, "uniform"), (3, "uniform"))
    for bits, name in configs:
        rows.append(
            ("%d-bit %s" % (bits, name), bits, "nf" if name == "NF" else "uniform")
        )

    for label, bits, table in rows:
        if bits is None:
            st, loss, bpp, sbytes = sd, base, 32.0, fp32_bytes
        else:
            st = quantize_state(sd, bits, table, BLOCK)
            t0 = time.perf_counter()
            loss = evaluate(build(st), val)
            dt = time.perf_counter() - t0
            sbytes = storage_bytes(sd, bits, table, BLOCK)
            bpp = sbytes * 8.0 / sum(v.numel() for v in sd.values())
            print(f"  ({label} eval {dt:.1f} s)")
        print(
            "%-18s %10.4f %+10.4f %10.2f %10.1f %8.2fx"
            % (
                label,
                loss,
                loss - base,
                bpp,
                sbytes / 2**20,
                fp32_bytes / sbytes if sbytes else 1.0,
            )
        )
    if QUICK:
        return
    print()
    # block-size sensitivity for NF2 (storage/quality trade-off)
    print("NF2 block 敏感度（block 越大存储越省、误差越大）：")
    for block in (32, 64, 128, 256):
        st = quantize_state(sd, 2, "nf", block)
        loss = evaluate(build(st), val)
        sbytes = storage_bytes(sd, 2, "nf", block)
        print(
            "  block %3d: loss %.4f (%+.4f)  %.2f bit/权重  %.1f MB"
            % (
                block,
                loss,
                loss - base,
                sbytes * 8.0 / sum(v.numel() for v in sd.values()),
                sbytes / 2**20,
            )
        )


if __name__ == "__main__":
    main()
