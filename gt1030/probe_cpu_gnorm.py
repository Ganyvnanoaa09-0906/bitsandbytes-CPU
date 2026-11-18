# -*- coding: utf-8 -*-
"""probe_cpu_gnorm.py — 用 CPU fp32 autograd 求"真"梯度范数，与核显 fp16 存储的范数和上报值对比。

结论要看三者的关系：
  A = CPU fp32 真梯度范数
  B = 核显 fp16 梯度缓冲里算出来的范数（host 侧 fp64 复算）
  C = 核显 Model.grad_norm() 上报值（= sqrt(sum) / CE_SCALE）
若 B ≈ A * CE_SCALE ⇒ 核显把梯度按 CE_SCALE(1024) 放大后存进 fp16（headroom 少 10 bit）；
若 B ≈ A ⇒ 那么 C 被错误地缩小了 1024 倍，clip/跳步判据全部失效。
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
print(f"[probe] ckpt={os.path.basename(CKPT)} tokens={tuple(tokens.shape)} CE_SCALE={MM.CE_SCALE}")

# ---------- 核显 ----------
m = MM.Model(CKPT)
loss_ipu = m.forward(np.ascontiguousarray(tokens.numpy()[:, :MM.T]))
m.backward(np.ascontiguousarray(tokens.numpy()[:, :MM.T]))
C = m.grad_norm()
B = 0.0
for blk in m.blocks:
    d = blk.dev
    d.activate()
    for k in ("Wqkv", "Wo", "W12", "W3"):
        p = blk.W[k]
        g = d.down(p.grad, (p.n,), np.float16).astype(np.float64)
        B += float(np.sum(g * g))
h = m.head
h.dev.activate()
gh = h.dev.down(h.W.grad, (h.W.n,), np.float16).astype(np.float64)
B += float(np.sum(gh * gh))
B = float(np.sqrt(B))

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
def _pick(o):
    if isinstance(o, (tuple, list)):
        cand = [x for x in o if hasattr(x, shape)]
        return cand[-1] if cand else _pick(o[-1])
    return o
logits = _pick(out)
shift = 1 if not getattr(cfg, "use_bos", False) else 0
lg = logits[:, :-1] if shift else logits
lg = lg.reshape(-1, lg.shape[-1]).float()
tg = (tokens[:, 1:] if shift else tokens).reshape(-1)
loss_cpu = torch.nn.functional.cross_entropy(lg, tg)
loss_cpu.backward()
sq = 0.0
for p in cm.parameters():
    if p.grad is not None:
        sq += float(p.grad.detach().double().pow(2).sum())
A = float(np.sqrt(sq))

print(f"  iGPU loss = {loss_ipu:.6f}   CPU loss = {float(loss_cpu):.6f}")
print(f"  A = CPU fp32 真梯度范数        = {A:.4f}")
print(f"  B = 核显 fp16 缓冲里的范数     = {B:.4f}")
print(f"  C = 核显 grad_norm() 上报值    = {C:.4f}")
print(f"  B / A = {B/max(A,1e-30):.4f}     C * CE_SCALE / A = {C*MM.CE_SCALE/max(A,1e-30):.4f}")
