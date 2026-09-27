# -*- coding: utf-8 -*-
"""test_temporal_only_lora.py — --lora_scope（只训时序层）的回归测试

为什么有这个文件:
    `--lora_scope temporal|spatial|all` 是 report §10.157 新加的开关。接线时它曾
    **静默失效**（apply_method 把 lora_scope 收进 **extra 后整个丢掉），
    而当时唯一的发现手段是端到端断言"期望的可训练参数量"。
    ⇒ 把这个断言固化成可重复运行的测试，而不是留在一次性的手工命令里。

风格对齐 `test_peft_backends.py`：独立脚本 + 退出码（本仓没有 pytest 配置）。

覆盖三层，从快到慢：
    [A] 分类逻辑（纯合成模块树，秒级，无权重、无 torch）
    [B] PEFT 注入 + restrict（小合成模型，仍然秒级）
    [C] 真实 AnimateDiff（需权重；用 BNB_TEST_REAL_MODEL=1 开启，约 40 s）

[A] 是关键：分类靠**模块名正则**，而实测时序模块叫 `motion_modules`
（**不含 `temporal`**，见 §10.149 / §10.156.4）。名字猜错的代价是"静默按空间处理"，
所以这里用真实命名结构做合成树，把那两个命名陷阱钉住。
"""
from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

FAILED = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag:4s}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)
    return cond


# ---------------------------------------------------------------------------
# [A] 分类逻辑：纯名字，不需要 torch
# ---------------------------------------------------------------------------
def section_a():
    print("=" * 74)
    print("[A] 分类逻辑（正则，无 torch）")
    print("=" * 74)
    from temporal_only_lora import _is_temporal, _is_spatial

    # 真实 AnimateDiff 路径（实测自 report §10.156.2）
    temporal_names = [
        "base_model.model.down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_q",
        "base_model.model.mid_block.motion_modules.0.transformer_blocks.0.attn2.to_out.0",
        "base_model.model.up_blocks.1.motion_modules.2.transformer_blocks.0.attn1.to_v",
    ]
    spatial_names = [
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q",
        "base_model.model.mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0",
        "base_model.model.up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v",
    ]
    for n in temporal_names:
        check("temporal: " + n.split(".")[-3] + "." + n.split(".")[-1], _is_temporal(n))
    for n in spatial_names:
        check("spatial : " + n.split(".")[-3] + "." + n.split(".")[-1],
              _is_spatial(n) and not _is_temporal(n))

    # 命名陷阱：时序模块名里**没有** temporal 字样，空间/时序叶子名**相同**
    check("trap: 'temporal' 不应是唯一依据",
          not any("temporal" in n for n in temporal_names))
    check("trap: 叶子名相同仍可区分",
          temporal_names[0].split(".")[-1] == spatial_names[0].split(".")[-1]
          and _is_temporal(temporal_names[0]) != _is_temporal(spatial_names[0]))

    # 时序不是空间的子集（互斥）
    for n in temporal_names:
        if _is_spatial(n):
            check("mutual exclusion for " + n, False)
            break
    else:
        check("temporal 与 spatial 互斥", True)


# ---------------------------------------------------------------------------
# [B] PEFT 注入 + restrict（小合成模型）
# ---------------------------------------------------------------------------
class _Block:
    pass


def _build_synthetic():
    """搭出与 AnimateDiff 同构的模块树：每层同时有 attentions（空间）与
    motion_modules（时序），两者内部都有 transformer_blocks.N.attn1.to_*，
    **叶子名完全相同**，正是 PEFT 无法按父路径区分的那种结构。"""
    import torch.nn as nn

    def attn_set():
        return nn.ModuleDict({
            "attn1": nn.ModuleDict({k: nn.Linear(8, 8, bias=False)
                                    for k in ("to_q", "to_k", "to_v")}),
        })

    class TB(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = nn.ModuleDict({k: nn.Linear(8, 8, bias=False)
                                        for k in ("to_q", "to_k", "to_v")})

    class AttnBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([TB()])

        def forward(self, x):
            # 必须**真的**把数据送过 to_q/to_k/to_v，否则 PEFT 注入的 LoRA 层
            # 不参与前向，反传拿不到梯度，"参数是否移动"就无从谈起。
            for tb in self.transformer_blocks:
                h = tb.attn1["to_q"](x) + tb.attn1["to_k"](x) + tb.attn1["to_v"](x)
                x = x + h
            return x

    class Level(nn.Module):
        def __init__(self, n_attn=2, n_motion=3):
            super().__init__()
            self.attentions = nn.ModuleList([AttnBlock() for _ in range(n_attn)])
            self.motion_modules = nn.ModuleList([AttnBlock() for _ in range(n_motion)])

        def forward(self, x):
            for m in self.attentions:
                x = m(x)
            for m in self.motion_modules:
                x = m(x)
            return x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_blocks = nn.ModuleList([Level(2, 3)])
            self.up_blocks = nn.ModuleList([Level(1, 2)])

        def forward(self, x):
            for lv in self.down_blocks:
                x = lv(x)
            for lv in self.up_blocks:
                x = lv(x)
            return x

    return Net()


def section_b():
    print()
    print("=" * 74)
    print("[B] PEFT 注入 + restrict（合成模型）")
    print("=" * 74)
    import torch
    from temporal_only_lora import classify_lora_targets, restrict_to_temporal

    net = _build_synthetic()
    base_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    from peft import LoraConfig, get_peft_model
    for p in net.parameters():
        p.requires_grad = False
    pnet = get_peft_model(net, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                          bias="none",
                                          target_modules=["to_q", "to_k", "to_v"]))
    tem, spa = classify_lora_targets(pnet)
    full = sum(p.numel() for p in pnet.parameters() if p.requires_grad)
    check("注入同时命中时序与空间", len(tem) > 0 and len(spa) > 0,
          f"temporal={len(tem)} spatial={len(spa)} trainable={full}")

    # temporal
    pnet, st = restrict_to_temporal(pnet, scope="temporal", verbose=True)
    tem2, spa2 = classify_lora_targets(pnet)
    tr_t = sum(p.numel() for p in pnet.parameters() if p.requires_grad)
    check("temporal: 只剩时序", len(tem2) == len(tem) and len(spa2) == 0,
          f"kept={len(tem2)} (expect {len(tem)})")
    check("temporal: 可训练量下降", tr_t < full, f"{tr_t} < {full}")

    # spatial（另一半）
    net2 = _build_synthetic()
    for p in net2.parameters():
        p.requires_grad = False
    p2 = get_peft_model(net2, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                         bias="none",
                                         target_modules=["to_q", "to_k", "to_v"]))
    t0, s0 = classify_lora_targets(p2)
    p2, _ = restrict_to_temporal(p2, scope="spatial", verbose=False)
    t1, s1 = classify_lora_targets(p2)
    tr_s = sum(p.numel() for p in p2.parameters() if p.requires_grad)
    check("spatial: 只剩空间", len(t1) == 0 and len(s1) == len(s0),
          f"kept={len(s1)} (expect {len(s0)})")

    # 分割自洽：temporal + spatial == all
    check("分割自洽: temporal + spatial == all", tr_t + tr_s == full,
          f"{tr_t} + {tr_s} = {tr_t + tr_s} vs all {full}")

    # 训练端到端：只有时序动
    net3 = _build_synthetic()
    for p in net3.parameters():
        p.requires_grad = False
    p3 = get_peft_model(net3, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                         bias="none",
                                         target_modules=["to_q", "to_k", "to_v"]))
    p3, _ = restrict_to_temporal(p3, scope="temporal", verbose=False)
    before = {}
    for n, m in p3.named_modules():
        if hasattr(m, "lora_B") and hasattr(m, "base_layer"):
            before[n] = (m.lora_B["default"].weight.detach().clone())
    opt = torch.optim.SGD([p for p in p3.parameters() if p.requires_grad], lr=0.1)
    for _ in range(2):
        x = torch.randn(2, 8)
        y = p3(x).sum()
        opt.zero_grad(set_to_none=True)
        y.backward()
        opt.step()
    moved_tem = moved_other = 0
    for n, m in p3.named_modules():
        if n in before:
            if (m.lora_B["default"].weight - before[n]).abs().max().item() > 0:
                from temporal_only_lora import _is_temporal
                if _is_temporal(n):
                    moved_tem += 1
                else:
                    moved_other += 1
    check("训练后时序参数确实变化", moved_tem > 0, f"{moved_tem} 个时序模块移动")
    check("训练后空间参数未变（应为 0）", moved_other == 0,
          f"{moved_other} 个非时序模块移动")


# ---------------------------------------------------------------------------
# [C] 真实模型（可选）
# ---------------------------------------------------------------------------
def section_c():
    if os.environ.get("BNB_TEST_REAL_MODEL") != "1":
        print()
        print("=" * 74)
        print("[C] 真实 AnimateDiff —— 跳过（设 BNB_TEST_REAL_MODEL=1 开启，约 40 s）")
        print("=" * 74)
        return
    print()
    print("=" * 74)
    print("[C] 真实 AnimateDiff")
    print("=" * 74)
    import copy
    import gc
    import torch
    from animatediff_train import load_animatediff_model, resolve_video_model
    from diffusion_backends import apply_method
    from temporal_only_lora import classify_lora_targets

    torch.set_num_threads(int(os.environ.get("THREADS", "6")))
    vm = os.environ.get(
        "BNB_VIDEO_MODEL",
        r"D:\work\textmodel\sd15_base\unet,"
        r"D:\work\textmodel\animatediff-motion-adapter-v1-5-2")
    u, a = resolve_video_model(vm)
    m = load_animatediff_model(u, a)

    res = {}
    for scope, expect in (("all", 2005504), ("temporal", 1208320), ("spatial", 797184)):
        # ⚠️ 不要每轮都 load_animatediff_model() —— 那会同时持有多个 1277M fp32 模型，
        #    在本机（15.4 GB）上实测触发 0xC0000005（与 report §10.155.3 同族的
        #    "多实例并存"问题）。改为载入一次 + deepcopy，并在每轮后显式释放。
        m2 = copy.deepcopy(m) if res else m
        s = apply_method("animatediff_lora", video_transformer=m2, rank=4, alpha=8,
                         dropout=0.0,
                         target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                         lora_scope=scope)
        tr = sum(p.numel() for p in s.extra["transformer"].parameters()
                 if p.requires_grad)
        res[scope] = tr
        check(f"real scope={scope}", tr == expect, f"{tr} (expect {expect})")
        del m2, s
        gc.collect()
    check("real 分割自洽", res["temporal"] + res["spatial"] == res["all"],
          f"{res['temporal']} + {res['spatial']} = {res['all']}")


if __name__ == "__main__":
    print("=" * 74)
    print("--lora_scope 回归测试")
    print("=" * 74)
    section_a()
    section_b()
    section_c()
    print()
    print("-" * 74)
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("ALL PASSED")
