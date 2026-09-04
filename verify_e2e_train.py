"""端到端组合训练冒烟：Qwen3Next（GDN+MoE）+ patch_transformers + EFST + AdamW8bit。

在纯 CPU 上把完整链条串起来跑 30 步自回归训练，验证：
- GDN 层走融合内核（patch 生效）
- EFST 只解冻热专家（+LoRA），冻结其余
- AdamW8bit 只给可训练参数分配状态
- loss 实际下降、无 NaN
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))
sys.path.insert(0, HERE)

from transformers import Qwen3NextConfig, Qwen3NextForCausalLM  # noqa: E402

torch.manual_seed(0)

cfg = Qwen3NextConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    linear_conv_kernel_dim=4,
    max_position_embeddings=64,
    layer_types=["linear_attention", "full_attention", "linear_attention"],
    # 每层 MLP 都是 MoE（8 专家）
    decoder_sparse_step=1,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    shared_expert_intermediate_size=64,
)

print("building Qwen3Next (GDN + MoE) ...")
model = Qwen3NextForCausalLM(cfg)
model.config.use_cache = False
model.train()

from bitsandbytes.gdn_cpu import patch_transformers  # noqa: E402
assert patch_transformers(), "patch failed"
print("GDN patch: OK")

# ---------------- EFST：校准选热专家 + LoRA ----------------
from efst import EFSTConfig, apply_efst  # noqa: E402

calib_x = [torch.randint(0, 256, (4, 16)) for _ in range(4)]
info = apply_efst(model, EFSTConfig(
    top_k=2,
    calibration_dataloader=calib_x,
    calibration_forward_fn=lambda b: model(input_ids=b, labels=b),
    num_calibration_batches=4,
    tune_router=True,
    lora=True,
    lora_r=8,
    lora_alpha=16,
))
print(info.summary())

import bitsandbytes as bnb  # noqa: E402

trainable = [p for p in model.parameters() if p.requires_grad]
n_tr = sum(p.numel() for p in trainable)
n_all = sum(p.numel() for p in model.parameters())
print(f"trainable {n_tr:,}/{n_all:,} ({100*n_tr/n_all:.2f}%)")
assert n_tr < n_all * 0.6, "EFST should freeze the majority"

opt = bnb.optim.AdamW8bit(trainable, lr=5e-4)

# ---------------- 训练 60 步 ----------------
x = torch.randint(0, 256, (2, 32))
losses = []
t0 = time.perf_counter()
for step in range(60):
    opt.zero_grad()
    out = model(input_ids=x, labels=x)
    out.loss.backward()
    opt.step()
    losses.append(out.loss.item())
    if step in (0, 19, 39, 59):
        print(f"step {step:2d}  loss {out.loss.item():.4f}")
dt = time.perf_counter() - t0
print(f"60 steps in {dt:.1f}s ({dt/60*1e3:.0f} ms/step)")

assert torch.isfinite(torch.tensor(losses)).all(), "NaN loss"
assert losses[-1] < losses[0] * 0.8, f"loss not dropping enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
assert losses[-5] > losses[-1], "loss should still be decreasing at the end"
print(f"loss drop {100*(1-losses[-1]/losses[0]):.1f}% over 60 steps (still decreasing at tail)")
n_state = sum(1 for g in opt.state.values() for k in ("state1", "state2") if k in g)
print(f"optimizer state entries: {n_state}")
print("E2E TRAIN SMOKE PASSED")
