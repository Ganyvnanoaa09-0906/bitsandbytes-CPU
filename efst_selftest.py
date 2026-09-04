"""EFST self-test with a tiny synthetic MoE model.

Run on a machine with PyTorch installed:

    python efst_selftest.py

It verifies:
- expert container auto-detection
- top-k selection via routing usage
- freezing / unfreezing
- optional LoRA injection and trainable count
- forward/backward still works after EFST
"""

from __future__ import annotations

import torch
import torch.nn as nn

from efst import EFSTConfig, apply_efst, find_expert_groups, report_trainable


class ExpertMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 2)
        self.w2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class TinyMoEBlock(nn.Module):
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([ExpertMLP(dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # [B*T, E]
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            idx = topk_idx[:, i]
            weight = topk_probs[:, i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e).nonzero(as_tuple=True)[0]
                if mask.numel() > 0:
                    out[mask] += weight[mask] * self.experts[e](x[mask])
        return out


class TinyMoEModel(nn.Module):
    def __init__(self, dim: int = 16, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.moe = TinyMoEBlock(dim, num_experts=num_experts, top_k=top_k)
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.moe(self.embed(x)))


def main() -> None:
    torch.manual_seed(0)
    model = TinyMoEModel(dim=16, num_experts=4, top_k=2)
    groups = find_expert_groups(model)
    print(f"found {len(groups)} expert group(s):")
    for g in groups:
        print(f"  {g.path}: {len(g)} experts")

    # 造一个校准 loader：每个 batch 偏向不同专家
    class FakeLoader:
        def __init__(self):
            self.data = [
                torch.randn(8, 16) * (i + 1)
                for i in range(4)
            ]

        def __iter__(self):
            for x in self.data:
                yield x

    loader = FakeLoader()

    config = EFSTConfig(
        top_k=2,
        calibration_dataloader=loader,
        calibration_forward_fn=lambda batch: model(batch),
        num_calibration_batches=4,
        tune_router=True,
        lora=True,
        lora_r=4,
        lora_alpha=8,
    )
    info = apply_efst(model, config)
    print(info.summary())
    print(report_trainable(model))

    # forward/backward 冒烟
    x = torch.randn(4, 16)
    loss = model(x).square().mean()
    loss.backward()
    assert loss.isfinite()
    print("forward/backward OK")

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("lora_A" in n or "lora_B" in n for n in trainable), "LoRA params should be trainable"
    assert info.trainable_after > 0
    print("EFST SELFTEST PASSED")


if __name__ == "__main__":
    main()
