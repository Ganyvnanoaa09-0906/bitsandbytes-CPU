# -*- coding: utf-8 -*-
"""probe_cpu_gnorm.py — CPU fp32 真梯度范数 vs 核显 fp16 缓冲范数 vs 上报值。

A = CPU fp32 真梯度范数
B = 核显 fp16 梯度缓冲的范数（host fp64 复算）
C = 核显 Model.grad_norm() 上报值（= sqrt(sum) / CE_SCALE）
"""
import os
import sys

import numpy as np
import torch

REPO = r"D:\work\bitsandbytes-CPU"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "gt1030"))
os.environ.setdefault("AR_ARENA_MB", "3200")
import ar_v2_multi as MM  # noqa: E402
import scripts.lib.small_image_model_v2 as M  # noqa: E402

TOK = r"D:\work\cloud_salvage\ScPeP7\tokens_V32.pt"
CKPT = sys.argv[1] if len(sys.argv) > 1 else r"F:\weibo_cos\ar_cpu2\ckpt_46000.pt"

tk = torch.load(TOK, map_location="cpu", weights_only=False)
tk = (tk.get("tokens") if isinstance(tk, dict) else tk).long()
rng = np.random.RandomState(0)
idx = rng.choice(len(tk), 2, replace=False)
tokens = tk[idx].long()
arr = np.ascontiguousarray(tokens.numpy()[:, :MM.T])
print("[probe] ckpt=%s tokens=%s CE_SCALE=%s" % (os.path.basename(CKPT), tuple(tokens.shape), MM.CE_SCALE))

# ---------- 核显 ----------
m = MM.Model(CKPT)
loss_ipu = m.forward(arr)
m.backward(arr)
C = m.grad_norm()
B2 = 0.0
for blk in m.blocks:
    d = blk.dev
    d.activate()
    for k in ("Wqkv", "Wo", "W12", "W3"):
        p = blk.W[k]
        g = d.down(p.grad, (p.n,), np.float16).astype(np.float64)
        B2 += float(np.sum(g * g))
h = m.head
h.dev.activate()
gh = h.dev.down(h.W.grad, (h.W.n,), np.float16).astype(np.float64)
B2 += float(np.sum(gh * gh))
B = float(np.sqrt(B2))

# ---------- CPU fp32 ----------
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
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
out = cm.forward_logits(tokens, None)


def pick_logits(o):
    """从 tuple/list 里挑出最后一个带 .shape 的张量。"""
    if isinstance(o, (tuple, list)):
        cand = [x for x in o if hasattr(x, "shape")]
        return cand[-1] if cand else pick_logits(o[-1])
    return o


logits = pick_logits(out)
use_bos = bool(getattr(cfg, "use_bos", False))
if use_bos:
    lg, tg = logits, tokens
else:
    lg, tg = logits[:, :-1], tokens[:, 1:]
lg2 = lg.reshape(-1, lg.shape[-1]).float()
loss_cpu = torch.nn.functional.cross_entropy(lg2, tg.reshape(-1))
loss_cpu.backward()
sq = 0.0
for p in cm.parameters():
    if p.grad is not None:
        sq += float(p.grad.detach().double().pow(2).sum())
A = float(np.sqrt(sq))

print("  iGPU loss = %.6f   CPU loss = %.6f" % (loss_ipu, float(loss_cpu)))
print("  A = CPU fp32 真梯度范数     = %.4f" % A)
print("  B = 核显 fp16 缓冲里的范数  = %.4f" % B)
print("  C = 核显 grad_norm() 上报值 = %.4f" % C)
print("  B / A = %.4f" % (B / max(A, 1e-30)))
print("  C * CE_SCALE / A = %.4f" % (C * MM.CE_SCALE / max(A, 1e-30)))
print("  B * (head 占比) 见下一行：head 的 B 分量 = %.4f" % float(np.sqrt(float(np.sum(gh * gh)))))
