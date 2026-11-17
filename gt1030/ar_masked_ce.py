# -*- coding: utf-8 -*-
"""ar_masked_ce.py — 用固定掩码评估 MaskGIT 的掩码交叉熵（与采样无关的客观指标）。

为什么需要：采样结果受 top-k/迭代调度影响，肉眼判读主观；掩码 CE 是同一把尺子——
拿固定种子生成同一批掩码，只算被遮位置的 CE。均匀分布时 CE ≈ ln(1024) = 6.931。

用法：
  python ar_masked_ce.py --ckpt F:\\weibo_cos\\_mgit_scr_b4\\ckpt_1500.pt \
      --tokens F:\\weibo_cos\\tokens_32x32_weibo.pt --n 512 --ratio 0.5 --threads 2
"""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--block", type=int, default=0, help=">0 时用 KxK 整块掩码")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--loop", type=int, default=0, help="循环轮数 L（0=最大值）")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfgd = ck["cfg"]
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    model = SmallARImageModelV2(cfg)
    sd = ck["model"]
    if any(k.startswith("loop_mods.") for k in sd):
        from scripts.lib.loop_transformer import LoopConfig
        lmax = 4
        for k, v in sd.items():
            if k.endswith("step_enc.table.weight") or k.endswith("cross_res.gate"):
                lmax = max(1, v.shape[0] - 1)
                break
        model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                                  loop_L=(a.loop or lmax), loop_L_max=lmax, loop_random=False,
                                  loop_cache="shared"))
    model.load_state_dict(sd)
    model.eval()
    mask_id = int(cfg.vocab_size - 1)
    V = mask_id
    T = int(cfg.total_tokens)
    G = int(round(T ** 0.5))

    toks = torch.load(a.tokens, map_location="cpu", weights_only=False)
    if isinstance(toks, dict):
        toks = toks.get("tokens", list(toks.values())[0])
    toks = toks.long().view(toks.shape[0], -1)[: a.n]

    g = torch.Generator().manual_seed(1234)
    tot, cnt, uniform = 0.0, 0, math.log(V)
    with torch.no_grad():
        for i in range(0, toks.shape[0], 16):
            x = toks[i:i + 16]
            B = x.shape[0]
            if a.block > 0 and G % a.block == 0:
                nb = G // a.block
                k = max(1, int(round(a.ratio * nb * nb)))
                bm = torch.zeros(B, nb * nb, dtype=torch.bool)
                for b in range(B):
                    perm = torch.randperm(nb * nb, generator=g)
                    bm[b, perm[:k]] = True
                mask = (bm.reshape(B, nb, nb).unsqueeze(2).unsqueeze(4)
                        .expand(B, nb, a.block, nb, a.block).reshape(B, T))
            else:
                mask = torch.rand(B, T, generator=g) < a.ratio
            if not bool(mask.any()):
                continue
            x_in = torch.where(mask, torch.full_like(x, mask_id), x)
            lg = model.forward_masked(x_in, None)[..., :V]
            tot += float(F.cross_entropy(lg[mask], x[mask])) * int(mask.sum())
            cnt += int(mask.sum())
    ce = tot / max(1, cnt)
    rep = dict(ckpt=a.ckpt.split("\\")[-2] + "/" + a.ckpt.split("\\")[-1], ratio=a.ratio,
               block=a.block, n=int(toks.shape[0]), masked_positions=cnt,
               ce=round(ce, 4), uniform_ce=round(uniform, 4),
               gap_to_uniform=round(ce - uniform, 4))
    print("  " + json.dumps({k: rep[k] for k in ("ckpt", "ratio", "block", "ce", "uniform_ce", "gap_to_uniform")},
                            ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
