# -*- coding: utf-8 -*-
"""Whole-checkpoint NF2 storage on the real 30M AR model: pack -> save -> load -> eval.

Acceptance:
  1. the packed file is ~12.7x smaller than the fp32 state dict;
  2. reloading the packed file reproduces the in-memory quantised tensors bit-for-bit;
  3. the reloaded model's eval loss equals the in-memory quantised eval loss.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "bitsandbytes"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
sys.stdout.reconfigure(encoding="utf-8")

from bitsandbytes import nfbit  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402

CKPT = r"D:\work\cloud_salvage\oWAZY8\ckpt_3000.pt"
TOKENS = r"D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt"
BLOCK = 64


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


def main():
    torch.set_num_threads(6)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if torch.is_tensor(v)}
    names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
    cfg = SmallImageConfigV2(
        **{k: v for k, v in (ck.get("cfg") or {}).items() if k in names}
    )
    val = torch.load(TOKENS, map_location="cpu", weights_only=False)[-128:].long()

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
    print("fp32 state dict: %.1f MB" % (fp32_bytes / 2**20))

    packed = nfbit.pack_state_dict(sd, bits=2, block=BLOCK, min_numel=4096)
    packed_bytes = nfbit.packed_state_bytes(packed)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model.nf2.pt")
        t0 = time.perf_counter()
        torch.save(
            {
                "__nfbit_meta__": packed["__nfbit_meta__"],
                **{k: v for k, v in packed.items() if k != "__nfbit_meta__"},
            },
            p,
        )
        save_s = time.perf_counter() - t0
        file_bytes = os.path.getsize(p)
        t0 = time.perf_counter()
        loaded = torch.load(p, map_location="cpu", weights_only=False)
        load_s = time.perf_counter() - t0
        unpacked = nfbit.unpack_state_dict(loaded)
    print(
        "packed payload: %.1f MB (%.2fx) | file on disk: %.1f MB (%.2fx) | save %.1fs load %.1fs"
        % (
            packed_bytes / 2**20,
            fp32_bytes / packed_bytes,
            file_bytes / 2**20,
            fp32_bytes / file_bytes,
            save_s,
            load_s,
        )
    )

    # bit-exact reload vs the in-memory quantisation
    in_memory = nfbit.unpack_state_dict(packed)
    mismatched = [
        k
        for k in in_memory
        if torch.is_tensor(in_memory[k]) and not torch.equal(in_memory[k], unpacked[k])
    ]
    print(
        "reload bit-exact over %d tensors: %s"
        % (len(unpacked), "YES" if not mismatched else "NO %s" % mismatched[:3])
    )

    t0 = time.perf_counter()
    loss_fp32 = evaluate(build(sd), val)
    loss_nf2 = evaluate(build(unpacked), val)
    print(
        "eval loss: fp32 %.4f | NF2-from-disk %.4f (%.1f s)"
        % (loss_fp32, loss_nf2, time.perf_counter() - t0)
    )


if __name__ == "__main__":
    main()
