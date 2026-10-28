# -*- coding: utf-8 -*-
"""test_efst_regressions.py — 回归测试：嵌套 MoE 块识别 + 3D 张量专家 + 非 MoE 安全。

这些是 2026-10-19 修复的内容：
  1. 标准 MoE 块（block 里同时有 gate 和 experts）此前被当成 2 专家容器，
     top-k 会选到 gate/experts，专家永远冻结不了。修复后应识别内层 experts 容器。
  2. 3D 张量专家（transformers 5.15+ Qwen3Next/Qwen3.5 风格：权重 [E, ...]）路径。
  3. 非 MoE 模型上 apply_efst 不再静默冻结全部参数，而是明确报错（可显式放行）。
"""
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import torch
import torch.nn

from scripts.lib.efst import EFSTConfig, apply_efst, find_expert_groups


# ---------------------------------------------------------------- case 1
class ExpertMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 2)
        self.w2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class MoEBlock(nn.Module):
    """标准结构：gate（router）+ experts（ModuleList），block 类型名含 Moe。"""

    def __init__(self, dim=32, num_experts=8, top_k=2):
        super().__init__()
        self.dim, self.num_experts, self.top_k = dim, num_experts, top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([ExpertMLP(dim) for _ in range(num_experts)])

    def forward(self, x):
        probs = torch.softmax(self.gate(x), -1)
        weights, indices = torch.topk(probs, self.top_k, -1)
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            idx = indices[:, slot]
            w = weights[:, slot].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        return out


class NestedMoEModel(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.moe = MoEBlock(dim)

    def forward(self, x):
        h = self.embed(x)
        return h + self.moe(h)


def case_nested_block():
    model = NestedMoEModel()
    groups = find_expert_groups(model)
    assert len(groups) == 1, [g.path for g in groups]
    g = groups[0]
    assert g.path == "moe.experts", g.path
    assert len(g) == 8, len(g)

    top_k = 2
    info = apply_efst(model, EFSTConfig(
        expert_indices=[0, 5], tune_router=True, lora=True, lora_r=8, lora_alpha=16))
    assert info.selected_experts == {"moe.experts": [0, 5]}, info.selected_experts
    # 未选中专家必须全冻结
    for e in (1, 2, 3, 4, 6, 7):
        assert all(not p.requires_grad for p in model.moe.experts[e].parameters()), e
    # 选中专家必须有可训练参数（LoRA）
    for e in (0, 5):
        assert any(p.requires_grad for p in model.moe.experts[e].parameters()), e
    # router 可训练
    assert all(p.requires_grad for p in model.moe.gate.parameters())
    # 前向/反向仍然可跑
    x = torch.randn(4, 32)
    loss = model(x).pow(2).mean()
    loss.backward()
    print("[PASS] case 1: 嵌套 MoE 块 -> 识别 moe.experts(8)，未选中专家全冻结，LoRA 可训，前后向 OK")


# ---------------------------------------------------------------- case 2
class TinyTensorExperts(nn.Module):
    """Qwen3Next/Qwen3.5 风格：专家权重是 [E, ...] 的大 Parameter，没有子模块。"""

    def __init__(self, hidden=16, inter=24, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = nn.SiLU()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden, 2 * inter) * 0.02)  # [E, in, out]: grouped-mm layout
        self.down_proj = nn.Parameter(torch.randn(num_experts, inter, hidden) * 0.02)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        mask = mask.permute(2, 1, 0)
        for e in range(self.num_experts):
            m = mask[e]
            if not bool(m.any()):
                continue
            pos, tok = torch.where(m)
            state = hidden_states[tok]
            gate, up = (state @ self.gate_up_proj[e]).chunk(2, -1)
            cur = (self.act_fn(gate) * up) @ self.down_proj[e]
            cur = cur * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur)
        return final


class TensorMoEModel(nn.Module):
    def __init__(self, hidden=16, inter=24, num_experts=4, top_k=2):
        super().__init__()
        self.hidden, self.num_experts, self.top_k = hidden, num_experts, top_k
        self.gate = nn.Linear(hidden, num_experts)
        self.experts = TinyTensorExperts(hidden, inter, num_experts)

    def forward(self, x):
        probs = torch.softmax(self.gate(x), -1)
        w, idx = torch.topk(probs, self.top_k, -1)
        return self.experts(x, idx, w)


def case_tensor_experts():
    model = TensorMoEModel()
    groups = find_expert_groups(model)
    assert len(groups) == 1, [g.path for g in groups]
    g = groups[0]
    assert g.path == "moe_placeholder" or g.path == "experts", g.path
    assert g.is_tensor and len(g) == 4, (g.is_tensor, len(g))
    info = apply_efst(model, EFSTConfig(expert_indices=[1, 3], tune_router=True, lora=True))
    assert info.selected_experts == {"experts": [1, 3]}, info.selected_experts
    # 拆成 ParameterList 后按专家行解冻：选中的行可训练，未选中的行冻结
    assert isinstance(model.experts.gate_up_proj, nn.ParameterList)
    for e in range(4):
        rows = model.experts.gate_up_proj[e]
        want = e in (1, 3)
        assert rows.requires_grad == want, (e, rows.requires_grad, want)
    # 前向/反向（自定义 forward 已被替换成逐专家版本）。
    # 随机路由不一定覆盖选中的专家，这里显式构造覆盖所有专家的 top_k，保证梯度检查确定。
    x = torch.randn(4, 16)
    idx = torch.tensor([[0, 1], [2, 3], [1, 3], [0, 2]])
    w = torch.full((4, 2), 0.5)
    loss = model.experts(x, idx, w).pow(2).mean()
    loss.backward()
    assert model.experts.gate_up_proj[1].grad is not None
    assert model.experts.gate_up_proj[3].grad is not None
    assert model.experts.gate_up_proj[0].grad is None
    assert model.experts.gate_up_proj[2].grad is None
    print("[PASS] case 2: 3D 张量专家 -> 按专家行解冻，选中行可训/有梯度，未选中行冻结")


# ---------------------------------------------------------------- case 3
def case_no_expert_model():
    plain = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
    assert find_expert_groups(plain) == []
    try:
        apply_efst(plain, EFSTConfig(lora=True))
    except ValueError as exc:
        assert "no MoE expert containers" in str(exc), str(exc)
    else:
        raise AssertionError("非 MoE 模型应报错，而不是静默冻结全部参数")
    # 显式放行时仍然可用（整模型冻结场景）
    info = apply_efst(plain, EFSTConfig(allow_no_experts=True, lora=False))
    assert info.trainable_after == 0, info.trainable_after
    print("[PASS] case 3: 非 MoE 模型默认报错，allow_no_experts=True 显式放行")


def case_tensor_usage_counts():
    """Regression: fused tensor-expert routing usage must not be all-zero.

    HF transformers 5.x Mixtral stores experts as 3D Parameters and calls
    experts(hidden, top_k_index, top_k_weights). collect_expert_usage keys tensor-expert
    counters by strings, so the int keys from top_k_index used to miss and every count
    stayed 0 -> select_top_experts always returned the first keys ([0,1]). This case pins
    the int/str normalization.
    """
    import torch.nn
    from bitsandbytes.efst import (collect_expert_usage, find_expert_groups,
                                   select_top_experts)

    class TensorExperts(nn.Module):
        def __init__(self, n=4, h=8, i=4):
            super().__init__()
            self.num_experts = n
            self.gate_up_proj = nn.Parameter(torch.randn(n, 2 * i, h))
            self.down_proj = nn.Parameter(torch.randn(n, h, i))

        def forward(self, hidden, top_k_index, top_k_weights):
            return hidden

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = TensorExperts()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Blk()

        def forward(self, top_k_index, top_k_weights):
            return self.block.experts(torch.zeros(4, 8), top_k_index, top_k_weights)

    model = Model()
    groups = find_expert_groups(model)
    assert len(groups) == 1 and groups[0].is_tensor, groups
    idx = torch.tensor([[0, 3], [0, 3], [1, 2], [3, 3]])
    weights = torch.ones(4, 2)
    usage = collect_expert_usage(model, [None], forward_fn=lambda _: model(idx, weights))
    path = groups[0].path
    counts = usage[path]
    expected = {0: 2, 1: 1, 2: 1, 3: 4}
    for k, v in expected.items():
        got = counts.get(k, counts.get(str(k), 0))
        assert got == v, (k, got, v, counts)
    selected = select_top_experts(model, [None], 1, forward_fn=lambda _: model(idx, weights))
    assert selected[path][0] in (3, "3"), selected
    print("[PASS] case 4: 融合张量专家的路由 usage 计数与 top-k 选择正确")


def main():
    torch.manual_seed(0)
    case_nested_block()
    case_tensor_experts()
    case_tensor_usage_counts()
    case_no_expert_model()
    print("\nEFST REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()
