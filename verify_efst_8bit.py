"""EFST + 8-bit 优化器组合验证：冻结参数不占优化器状态，训练收敛正常。"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))  # 本地 bitsandbytes
sys.path.insert(0, HERE)  # efst.py

import bitsandbytes as bnb  # noqa: E402
from efst import EFSTConfig, apply_efst  # noqa: E402


class TinyMoE(nn.Module):
    def __init__(self, dim=32, num_experts=8, top_k=2):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            idx = topk_idx[:, i]
            w = topk_probs[:, i].unsqueeze(-1)
            for e in range(len(self.experts)):
                m = (idx == e).nonzero(as_tuple=True)[0]
                if m.numel():
                    out[m] += w[m] * self.experts[e](x[m])
        return out


torch.manual_seed(0)
model = nn.Sequential(nn.Linear(32, 32), TinyMoE(32), nn.Linear(32, 1))

loader = [torch.randn(16, 32) * (i + 1) for i in range(4)]
info = apply_efst(model, EFSTConfig(
    top_k=2,
    calibration_dataloader=loader,
    calibration_forward_fn=lambda b: model(b),
    num_calibration_batches=4,
    tune_router=True,
    lora=True,
    lora_r=8,
    lora_alpha=16,
))
print(info.summary())

# 8-bit 优化器只喂可训练参数
trainable = [p for p in model.parameters() if p.requires_grad]
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in trainable)
opt = bnb.optim.AdamW8bit(trainable, lr=1e-3)

x = torch.randn(64, 32)
w = torch.randn(32)
y = x @ w
losses = []
for step in range(80):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x).squeeze(-1), y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

print(f"trainable {n_train:,}/{n_total:,} ({100*n_train/n_total:.2f}%)")
print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}  (drop {100*(1-losses[-1]/losses[0]):.0f}%)")
assert losses[-1] < losses[0] * 0.7, "EFST+8bit barely moved"
n_state = sum(1 for g in opt.state.values() for k in ("state1", "state2") if k in g)
print(f"optimizer state entries: {n_state} (expect ~2x trainable params, no frozen params)")
print("EFST + AdamW8bit COMBO PASSED")
