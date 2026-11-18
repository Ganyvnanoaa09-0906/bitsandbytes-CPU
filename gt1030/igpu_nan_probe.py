# -*- coding: utf-8 -*-
"""igpu_nan_probe.py — 定位核显 AR 训练"NAN/gnorm 飙升"的来源。

核心怀疑：**梯度缓冲是 fp16**（Param.grad = dev.buf(n*2)，backward 全走 hgemm 写 fp16）。
fp16 上限 65504、最小正规数 6e-5，所以：
  (a) 单个梯度元素 >65504 → inf → gnorm=inf → 整步被跳；
  (b) 小梯度（<6e-5）下溢为 0/次正规 → 梯度被系统性丢掉；
  (c) 两者都会让"有效梯度方向"越来越偏，表现为 loss 漂移 + gnorm 慢慢爬。

本脚本做三件事：
  1) 取一个 ckpt、一批 token，跑 forward/backward；
  2) 把每层 fp16 梯度下载下来做**高精度统计**（inf 数、饱和数、下溢数、max、fp64 平方和）；
  3) 用同样权重/同样 batch 在同一台机器上跑 **CPU fp32 autograd** 求"真"梯度范数作参照。
"""
import argparse
import os
import sys

import numpy as np

REPO = r"D:\work\bitsandbytes-CPU"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("AR_ARENA_MB", "3200")

sys.path.insert(0, os.path.join(REPO, "gt1030"))
import ar_v2_multi as MM  # noqa: E402

TOK = r"D:\work\cloud_salvage\ScPeP7\tokens_V32.pt"
FP16_MAX = 65504.0
FP16_MINNORM = 6.104e-5


def stat_fp16(buf, dev, n):
    g = dev.down(buf, (n,), np.float16)
    dev.finish()
    a = g.astype(np.float32)
    finite = np.isfinite(a)
    bad = int((~finite).sum())
    sat = int((np.abs(a[finite]) >= FP16_MAX * 0.999).sum())
    sub = int(((a[finite] != 0) & (np.abs(a[finite]) < FP16_MINNORM)).sum())
    zero = int((a == 0).sum())
    ss = float(np.sum(a[finite].astype(np.float64) ** 2))
    return dict(n=int(n), bad=bad, sat=sat, subnormal=sub, zero=zero,
                maxabs=float(np.abs(a[finite]).max()) if finite.any() else float("nan"),
                sumsq_f64=ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"F:\weibo_cos\ar_cpu2\ckpt_46000.pt")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    tk = torch.load(TOK, map_location="cpu", weights_only=False)
    tk = (tk.get("tokens") if isinstance(tk, dict) else tk).long()
    rng = np.random.RandomState(a.seed)
    idx = rng.choice(len(tk), a.batch, replace=False)
    tokens = np.ascontiguousarray(tk[idx].numpy()[:, :MM.T])
    print(f"[probe] ckpt={os.path.basename(a.ckpt)} batch={a.batch} tokens={tokens.shape}", flush=True)

    model = MM.Model(a.ckpt)
    loss = model.forward(tokens)
    model.backward(tokens)
    gn_dev = model.grad_norm()
    print(f"[probe] loss={loss:.6f}  device grad_norm={gn_dev:.4f}", flush=True)

    rows = []
    ss_total = 0.0
    for bi, blk in enumerate(model.blocks):
        d = blk.dev
        d.activate()
        for k in ("Wqkv", "Wo", "W12", "W3"):
            p = blk.W[k]
            st = stat_fp16(p.grad, d, p.n)
            st.update(layer=bi, name=k, kind="matrix")
            rows.append(st)
    head = model.head
    st = stat_fp16(head.W.grad, head.dev, head.W.n)
    st.update(layer="head", name="W", kind="matrix")
    rows.append(st)

    for r in rows:
        ss_total += r["sumsq_f64"]
    print("\n=== fp16 梯度矩阵逐层统计（inf/饱和/下溢）===")
    print(f"{'layer':>5} {'name':>5} {'n':>10} {'inf':>6} {'饱和':>6} {'次正规':>7} {'零':>7} {'max|g|':>10}")
    for r in rows:
        print(f"{str(r['layer']):>5} {r['name']:>5} {r['n']:>10} {r['bad']:>6} {r['sat']:>6} "
              f"{r['subnormal']:>7} {r['zero']:>7} {r['maxabs']:>10.4g}")

    tot_bad = sum(r["bad"] for r in rows)
    tot_sat = sum(r["sat"] for r in rows)
    tot_sub = sum(r["subnormal"] for r in rows)
    tot_n = sum(r["n"] for r in rows)
    print(f"\n  矩阵梯度合计：{tot_n} 个元素，inf {tot_bad}、饱和 {tot_sat}、次正规 {tot_sub}")
    print(f"  仅矩阵部分的 fp64 平方和 = {ss_total:.4f} ⇒ 范数 {np.sqrt(ss_total):.4f}")
    print(f"  设备报的 grad_norm（含小向量 + fp16 累加） = {gn_dev:.4f}")

    # CPU fp32 参照
    print("\n=== CPU fp32 autograd 参照 ===", flush=True)
    sys.path.insert(0, REPO)
    import scripts.lib.small_image_model_v2 as M
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    keep = set(M.SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = M.SmallImageConfigV2(**{k: v for k, v in ck["cfg"].items() if k in keep})
    cm = M.SmallARImageModelV2(cfg)
    sd = ck["model"]
    if any(k.startswith("loop_mods.") for k in sd):
        from scripts.lib.loop_transformer import LoopConfig
        mx = 4
        for k, v in sd.items():
            if k.endswith("step_enc.table.weight"):
                mx = v.shape[0]
                break
        cm.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                               loop_L=max(1, mx - 1), loop_L_max=max(1, mx - 1),
                               loop_random=False, loop_cache="shared"))
    cm.load_state_dict(sd)
    cm.train()
    t = torch.from_numpy(tokens).long()
    out = cm(t)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    loss_cpu = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), t.reshape(-1))
    loss_cpu.backward()
    sq = 0.0
    for p in cm.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().double().pow(2).sum())
    print(f"  CPU loss={float(loss_cpu):.6f}  CPU fp32 grad_norm={np.sqrt(sq):.4f}")
    print(f"  iGPU loss={loss:.6f}  iGPU 报的 grad_norm={gn_dev:.4f}")
    print("\n判读：若 iGPU 报的范数 ≫ CPU fp32 范数，且逐层统计里出现 inf/饱和，"
          "则根因是【梯度以 fp16 存储/累加】；若两者接近而只有 loss 漂移，则是优化器/数据侧问题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
