# -*- coding: utf-8 -*-
"""ar_ce_dispersion.py — 单样本 CE 的离散度：用来判断"训练曲线上的 loss 尖峰"是噪声还是不稳定。

思路：训练每步只看到 B=2 张图，所以打印出来的 loss 是 2 个样本 CE 的均值。
若单样本 CE 的标准差是 σ1，则 B=2 时每步 loss 的标准差应为 σ1/√2。
本脚本在留出集上**逐样本**算 CE，得到 σ1，再与训练日志的逐步 std 对比：
  - 两者一致 ⇒ 曲线波动就是采样噪声（不是数值不稳定，无需"修"）；
  - 训练 std 明显更大 ⇒ 真的有问题（需要查数值/优化器）。

用法：
  python ar_ce_dispersion.py --ckpt F:\\weibo_cos\\_from76000\\ckpt_76400.pt \
      --tokens F:\\weibo_cos\\tokens_32x32_weibo.pt --n 256 --threads 2
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--n", type=int, default=256)
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

    toks = torch.load(a.tokens, map_location="cpu", weights_only=False)
    if isinstance(toks, dict):
        toks = toks.get("tokens", list(toks.values())[0])
    toks = toks.long().view(toks.shape[0], -1)
    val = toks[-a.n:]
    V = int(cfg.vocab_size)

    per = []
    with torch.no_grad():
        for i in range(0, val.shape[0], 8):
            x = val[i:i + 8]
            B, T = x.shape
            o = model.forward_logits(x, None)
            gg = o[1]
            lg = gg[0] if isinstance(gg, tuple) else gg     # 与 train_ar_v2.val_loss 一致
            if getattr(cfg, "use_bos", False):
                src, tgt = lg, x
            else:
                src, tgt = lg[:, :-1], x[:, 1:]             # use_bos=False：移位预测
            ce = F.cross_entropy(src.reshape(-1, src.shape[-1]), tgt.reshape(-1),
                                 reduction="none")
            per.append(ce.reshape(tgt.shape[0], tgt.shape[1]).mean(1).numpy())
    per = np.concatenate(per)
    s1 = float(np.std(per, ddof=1))
    res = dict(ckpt=a.ckpt.split("\\")[-2] + "/" + a.ckpt.split("\\")[-1], n=int(len(per)),
               per_sample_ce_mean=round(float(per.mean()), 4),
               per_sample_ce_std=round(s1, 4),
               predicted_step_std=dict(b1=round(s1, 4), b2=round(s1 / math.sqrt(2), 4),
                                       b4=round(s1 / 2.0, 4), b8=round(s1 / math.sqrt(8), 4)))
    print("  " + json.dumps(res, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
