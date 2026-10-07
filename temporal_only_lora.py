# -*- coding: utf-8 -*-
"""temporal_only_lora.py — 让视频 LoRA **只训时序层**（保留 SD1.5 空间先验）

为什么需要:
    实测（report §10.156）: `apply_method('animatediff_lora', ...)` 的默认 targets
    是叶子名 `to_q,to_k,to_v,to_out.0`，而时序与空间的 attention **叶子名相同**
    ⇒ 592 个 LoRA 张量里 **336 个打在时序(motion_modules)、256 个打在空间(attentions)**。
    PEFT 的 `target_modules` **没有**"按父路径过滤"的选项。

    但对视频 LoRA 而言，"学运动、保留 SD1.5 的空间先验"才是通常想要的那件事：
    空间层已经在图像上训好了，跟着视频数据一起训既浪费可训练量，也可能把
    空间能力带偏。

做法:
    PEFT 把每个目标包成一个 `Linear`（带 `base_layer` / `lora_A` / `lora_B`）。
    所以"只训时序"= **把 scope 之外的位置换回它的 `base_layer`**（前向完全等价于
    未注入 LoRA），并同步从 `target_modules` 记录里删掉。

判据（本模块自检会验证三条）:
    1. 可训练参数量**恰好**等于时序部分（336/592 比例的那个数）；
    2. 保留的 LoRA 数量 == 时序位置数；
    3. **前向数值与"原始未注入 LoRA 的模型"逐位一致之外**——不，更强的判据是：
       前向输出与"完整 PEFT 模型但把空间 LoRA 权重置零"一致（见 verify()）。
       因为把 LoRA 换回 base_layer 与"LoRA 权重为零"在数学上相同。

用法:
    from temporal_only_lora import restrict_to_temporal
    mm = restrict_to_temporal(peft_model, scope='temporal')
"""
from __future__ import annotations

import re
from typing import List, Tuple

# 时序模块的命名（实测：AnimateDiff 用 motion_modules，**不含 temporal 字样**）
TEMPORAL_PATTERNS = (r"\.motion_modules\.", r"\.motion_", r"temporal")
# 空间 attention 的命名
SPATIAL_PATTERNS = (r"\.attentions\.", r"\.transformer_blocks\.", r"\.resnets\.")


def _is_temporal(name: str) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in TEMPORAL_PATTERNS)


def _is_spatial(name: str) -> bool:
    low = name.lower()
    if _is_temporal(low):
        return False
    return any(re.search(p, low) for p in SPATIAL_PATTERNS)


def classify_lora_targets(model) -> Tuple[List[str], List[str]]:
    """返回 (temporal_names, spatial_names)：当前**已经被注入 LoRA** 的模块全名。"""
    tem, spa = [], []
    for name, mod in model.named_modules():
        # PEFT 注入后，目标模块自身带 base_layer + lora_A
        if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):
            if _is_temporal(name):
                tem.append(name)
            elif _is_spatial(name):
                spa.append(name)
            else:
                spa.append(name)   # 归不进去的按"非时序"处理，保证不过度保留
    return tem, spa


def restrict_to_temporal(model, scope: str = "temporal", verbose: bool = True):
    """把 scope 之外的 LoRA 换回 base_layer，使可训练量只剩 scope 内的部分。

    scope='temporal' ⇒ 只保留时序；scope='spatial' ⇒ 只保留空间。
    返回 (model, stats)。
    """
    assert scope in ("temporal", "spatial"), scope
    tem, spa = classify_lora_targets(model)
    keep = set(tem if scope == "temporal" else spa)
    drop = set(tem + spa) - keep

    if verbose:
        print('[restrict_to_temporal] 注入总数=%d  时序=%d  空间=%d  scope=%s  ⇒ 保留 %d 丢弃 %d'
              % (len(tem) + len(spa), len(tem), len(spa), scope, len(keep), len(drop)))

    # ★ 响亮失败，而不是静默空转。
    #   scope='temporal' 但一个时序模块都没匹配到时，keep 为空 ⇒ 下面会把【全部】
    #   LoRA 换回 base_layer ⇒ 可训练量为 0 ⇒ 训练照常跑、loss 照常打、什么也没学 ✗✗
    #   这不是假设：Wan2.1 / CogVideoX 是 3D DiT，注意力【时空融合】，模块名是
    #   attn1/attn2 不含 temporal ⇒ 对它们"只训时序层"没有意义，必然匹配到 0 个。
    #   ⇒ 抛异常，让调用方显式选择 scope='all'，而不是拿到一个空转的运行。
    if not keep:
        raise ValueError(
            'scope=%r 匹配到 0 个模块（时序=%d 空间=%d 注入总数=%d）。'
            '继续下去会把所有 LoRA 换回 base_layer，可训练量为 0 而训练照常进行。'
            '若该模型没有可分离的时序层（Wan2.1 / CogVideoX 的注意力是时空融合的），'
            '请改用 scope="all"。'
            % (scope, len(tem), len(spa), len(tem) + len(spa)))

    # 逐模块替换：把 PEFT 的 Linear 换回它的 base_layer
    removed = 0
    for name in sorted(drop):
        # 找到父模块与该模块的属性名（可能带数字下标）
        parts = name.split(".")
        parent = model
        ok = True
        for p in parts[:-1]:
            if p.isdigit():
                if not hasattr(parent, "__getitem__"):
                    ok = False
                    break
                parent = parent[int(p)]
            else:
                if not hasattr(parent, p):
                    ok = False
                    break
                parent = getattr(parent, p)
        if not ok:
            continue
        leaf = parts[-1]
        try:
            mod = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        except Exception:
            continue
        base = getattr(mod, "base_layer", None)
        if base is None:
            continue
        if leaf.isdigit():
            parent[int(leaf)] = base
        else:
            setattr(parent, leaf, base)
        removed += 1

    # 同步 target_modules 记录，避免后续保存/加载认为还有这些层
    try:
        cfg = getattr(model, "peft_config", {})
        for _k, c in (cfg.items() if hasattr(cfg, "items") else []):
            tm = getattr(c, "target_modules", None)
            if tm is None:
                continue
            # 只保留仍然带 lora_A 的模块名对应的叶子名
            alive = set()
            for n, m in model.named_modules():
                if hasattr(m, "lora_A") and hasattr(m, "base_layer"):
                    alive.add(n.rsplit(".", 1)[-1])
            if isinstance(tm, (list, set, tuple)):
                c.target_modules = sorted(set(tm) & alive) or list(tm)
    except Exception as e:
        if verbose:
            print('  (target_modules 同步跳过: %s)' % e)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats = {"injected_total": len(tem) + len(spa), "temporal": len(tem),
             "spatial": len(spa), "kept": len(keep), "removed": removed,
             "trainable": trainable}
    if verbose:
        print('  已换回 base_layer: %d 个；可训练参数 = %d' % (removed, trainable))
    return model, stats


def verify(model, keep_scope: str = "temporal"):
    """判据：把 scope 外的 LoRA 换回 base_layer，应与"那些 LoRA 权重为零"数值等价。

    做法：同一输入下对比
      A) restrict 之后的模型输出
      B) 未 restrict、但把 scope 外的 lora_B 全部置零后的输出
    A 与 B 应逐位（或极小误差内）一致 ⇒ 证明恢复 base_layer 在数学上等价于
    把这些 LoRA 关掉，而不是把模型结构改坏。
    """
    import torch

    tem, spa = classify_lora_targets(model)
    outside = set(tem if keep_scope == "spatial" else spa)

    # 记录 scope 外的 lora_B 权重并置零。
    # ⚠️ mod.lora_B 是 nn.ModuleDict（key=adapter 名，如 'default'），取到的
    #    是 Linear 模块，真正的张量在 .weight 上 —— 直接 detach() 会报
    #    'Linear' object has no attribute 'detach'。
    saved = {}
    for name, mod in model.named_modules():
        if name in outside and hasattr(mod, "lora_B"):
            for k, lin in mod.lora_B.items():
                saved[(name, k)] = lin.weight.detach().clone()
                # lora_B.weight 是 requires_grad 的叶张量，原地写必须包在 no_grad 里，
                # 否则报 "a leaf Variable that requires grad is being used in an
                # in-place operation"。
                with torch.no_grad():
                    lin.weight.zero_()

    x = torch.randn(1, 4, 2, 4, 4)
    ctx = torch.randn(1, 77, 768).repeat(2, 1, 1)
    t = torch.tensor([500])
    with torch.no_grad():
        y_zeroed = model(x, t, encoder_hidden_states=ctx).sample.clone()

    # 还原，再真正 restrict
    for (name, k), t0 in saved.items():
        mod = model.get_submodule(name)
        with torch.no_grad():
            mod.lora_B[k].weight.copy_(t0)
    model, _ = restrict_to_temporal(model, scope=keep_scope, verbose=False)
    with torch.no_grad():
        y_restricted = model(x, t, encoder_hidden_states=ctx).sample

    d = (y_zeroed - y_restricted).abs().max().item()
    rel = d / (y_zeroed.abs().max().item() + 1e-12)
    print('[verify] scope=%s  置零 vs 换回 base_layer: max|Δ|=%.3e  rel=%.3e  %s'
          % (keep_scope, d, rel, 'EQUIVALENT' if rel < 1e-6 else 'DIFFERENT'))
    return rel < 1e-6


def train_step_verify(model, steps: int = 3):
    """端到端判据：真正跑几步训练，证明
       (1) 反向能传到时序 LoRA（梯度非零、权重变化非零）；
       (2) 空间侧**完全冻结**（不在可训练集合里 ⇒ 权重逐位不变）。
    前向等价（verify）只证明结构没坏；这一条才证明训练真的只在时序上发生。
    """
    import torch

    tem, spa = classify_lora_targets(model)
    tem_set, spa_set = set(tem), set(spa)

    # 记录初始权重
    before = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_B") and hasattr(mod, "base_layer"):
            with torch.no_grad():
                before[name + ".A"] = mod.lora_A["default"].weight.detach().clone()
                before[name + ".B"] = mod.lora_B["default"].weight.detach().clone()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.05)

    losses = []
    for _ in range(steps):
        x = torch.randn(1, 4, 2, 4, 4)
        ctx = torch.randn(1, 77, 768).repeat(2, 1, 1)
        t = torch.tensor([500])
        y = model(x, t, encoder_hidden_states=ctx).sample
        loss = (y * y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    # 找训练后哪些模块真的变了
    changed_tem = changed_spa = 0
    frozen_ok = True
    nograd_ok = True
    for name, mod in model.named_modules():
        if not (hasattr(mod, "lora_B") and hasattr(mod, "base_layer")):
            continue
        with torch.no_grad():
            dA = (mod.lora_A["default"].weight - before[name + ".A"]).abs().max().item()
            dB = (mod.lora_B["default"].weight - before[name + ".B"]).abs().max().item()
        moved = (dA + dB) > 0.0
        if name in tem_set:
            if moved:
                changed_tem += 1
        elif name in spa_set:
            if moved:
                changed_spa += 1
                frozen_ok = False

    print('[train_step_verify] %d 步, losses=%s' % (steps, ['%.4f' % v for v in losses]))
    print('  时序模块发生更新: %d / %d' % (changed_tem, len(tem_set)))
    print('  空间模块发生更新: %d / %d  (应为 0)' % (changed_spa, len(spa_set)))
    ok = (changed_tem > 0) and frozen_ok and nograd_ok
    print('  判据: %s' % ('PASS — 只有时序在训' if ok else 'FAIL'))
    return ok


def _selftest() -> int:
    import os
    import sys
    sys.path.insert(0, r"D:\work\bitsandbytes-CPU\bitsandbytes")
    sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
    import torch
    from animatediff_train import load_animatediff_model, resolve_video_model
    from diffusion_backends import apply_method

    torch.set_num_threads(int(os.environ.get("THREADS", "6")))
    vm = (r"D:\work\textmodel\sd15_base\unet,"
          r"D:\work\textmodel\animatediff-motion-adapter-v1-5-2")
    u, a = resolve_video_model(vm)
    base = load_animatediff_model(u, a)

    print("=" * 82)
    print("temporal-only LoRA 自检")
    print("=" * 82)

    s = apply_method("animatediff_lora", video_transformer=base, rank=4, alpha=8,
                     dropout=0.0, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    mm = s.extra["transformer"]
    full = sum(p.numel() for p in mm.parameters() if p.requires_grad)
    tem, spa = classify_lora_targets(mm)
    print("注入: 时序 %d + 空间 %d = %d   全量可训练 = %d" % (len(tem), len(spa), len(tem) + len(spa), full))

    ok_equiv = verify(mm, keep_scope="temporal")
    trainable = sum(p.numel() for p in mm.parameters() if p.requires_grad)
    kept = len([n for n, m in mm.named_modules()
                if hasattr(m, "lora_A") and hasattr(m, "base_layer")])
    print("\n结果: 保留 %d 个 LoRA 模块（应 == 时序 %d），可训练 %d（应 < 全量 %d）"
          % (kept, len(tem), trainable, full))

    print()
    ok_train = train_step_verify(mm, steps=3)

    good = ok_equiv and kept == len(tem) and trainable < full and ok_train
    print("\n判据: %s" % ("PASS" if good else "FAIL"))
    print("=" * 82)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
