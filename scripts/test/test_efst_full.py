# -*- coding: utf-8 -*-
"""test_efst_full.py — EFST 完整测试（MoE 前向/反向收敛 + top_k 选专家 + 非MoE fallback + 参数节省）"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys
os.environ["OMP_NUM_THREADS"] = "6"; os.environ["MKL_NUM_THREADS"] = "6"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))
import torch
import torch.nn
from scripts.lib.efst import EFSTConfig, apply_efst, find_expert_groups, report_trainable

class ExpertMLP(nn.Module):
    def __init__(self, dim): super().__init__(); self.w1 = nn.Linear(dim, dim*2); self.w2 = nn.Linear(dim*2, dim); self.act = nn.GELU()
    def forward(self, x): return self.w2(self.act(self.w1(x)))
class TinyMoEBlock(nn.Module):
    def __init__(self, dim=32, num_experts=8, top_k=2):
        super().__init__(); self.dim=dim; self.num_experts=num_experts; self.top_k=top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([ExpertMLP(dim) for _ in range(num_experts)])
    def forward(self, x):
        logits = self.gate(x); probs = torch.softmax(logits,-1)
        tp, ti = torch.topk(probs, self.top_k, -1)
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            idx = ti[:,i]; wgt = tp[:,i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx==e)
                if mask.any(): out[mask] += wgt[mask] * self.experts[e](x[mask])
        return out
class TinyMoEModel(nn.Module):
    def __init__(self, dim=32, num_experts=8, top_k=2):
        super().__init__(); self.embed = nn.Linear(dim,dim); self.moe = TinyMoEBlock(dim,num_experts,top_k)
    def forward(self, x): return self.embed(x) + self.moe(self.embed(x))

results = []
def check(name, cond, msg=""):
    r = "PASS" if cond else "FAIL"; results.append((name,r)); print(f"[{r}] {name} {msg}")

# 1) 识别专家组
m = TinyMoEModel()
groups = find_expert_groups(m)
check("find_expert_groups 识别", len(groups) > 0, f"{[(g.path, len(g)) for g in groups]}")

# 2) 用校准数据 top_k 自动选专家（官方路径：efst_selftest 同款）
class FakeLoader:
    def __init__(self):
        self.data = [torch.randn(8,32)*(i+1) for i in range(4)]
    def __iter__(self):
        for x in self.data: yield x
loader = FakeLoader()
n_before = sum(p.numel() for p in m.parameters() if p.requires_grad)
cfg = EFSTConfig(top_k=3, calibration_dataloader=loader,
                 calibration_forward_fn=lambda batch: m(batch),
                 num_calibration_batches=4, tune_router=True, lora=True, lora_r=8, lora_alpha=16)
info = apply_efst(m, cfg)
n_after = sum(p.numel() for p in m.parameters() if p.requires_grad)
check("apply_efst 节省参数(top_k)", 0 < n_after < n_before, f"{n_before}->{n_after} ({n_after/n_before*100:.1f}%)")

# 3) 前向+反向+优化 loss 收敛
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
x = torch.randn(8,32)
loss0 = m(x).pow(2).mean().item()
for _ in range(3):
    opt.zero_grad(); loss = m(x).pow(2).mean(); loss.backward(); opt.step()
loss1 = m(x).pow(2).mean().item()
check("MoE 优化 loss 下降", loss1 < loss0, f"{loss0:.4f}->{loss1:.4f}")

# 4) 冻结检查——嵌套 MoE 块结构（block 里同时有 gate 和 experts）曾是已知 bug：
#    EFST 会把 block 当成 2 专家容器，导致专家永远冻结不了。修复后应识别内层
#    `moe.experts`（8 专家），非选中专家全冻结、选中专家只有 LoRA 可训。
print(f"[diag] selected={getattr(info, 'selected_experts', None)}")
assert len(info.groups) == 1 and info.groups[0].path == "moe.experts", \
    [(g.path, len(g)) for g in info.groups]
selected = set(info.selected_experts["moe.experts"])
trainable_experts = [e for e in range(8)
                     if any(p.requires_grad for p in m.moe.experts[e].parameters())]
check("嵌套块结构下仅选中专家可训",
      set(trainable_experts) == {int(s) for s in selected},
      f"trainable={trainable_experts} selected={sorted(selected)}")
# 5) 非 MoE 模型安全
plain = nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Linear(32,32))
g2 = find_expert_groups(plain)
check("非MoE 无专家组", len(g2)==0, f"groups={len(g2)}")

print("\n=== 结果 ===")
fails = [n for n,r in results if r != "PASS"]
for n,r in results: print(f"  {r:4s} {n}")
print(f"{len(results)-len(fails)}/{len(results)} 通过")
raise SystemExit(1 if fails else 0)
