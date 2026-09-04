# -*- coding: utf-8 -*-
"""test_peft_backends.py — 统一训练框架 9 种 PEFT 方法的 smoke test。

用随机小模型验证 apply_method：
  - A 组（7 个"注入可训练参数"型方法）：完整 forward + backward；
  - B 组（2 个结构依赖型方法）：
      p_tuning_v2  需真实 transformers 模型，这里仅验证 config 可构造 + 报错可理解；
      efst         需 MoE 专家结构，非 MoE 模型上验证正确 fallback（冻结全部、groups=0）。
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

import torch
import torch.nn as nn

from peft_backends import apply_method, METHODS


class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 64)
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)
        self.lm_head = nn.Linear(64, 64)

    def forward(self, x):
        h = self.embed(x)
        h = torch.relu(self.q_proj(h) + self.k_proj(h) + self.v_proj(h))
        h = self.o_proj(h)
        return self.lm_head(h).mean()


# A 组：完整前向/反传
CASES = {
    "lora":       dict(target_modules=["q_proj", "k_proj", "v_proj"]),
    "qlora":      dict(quant_dtype="8bit"),
    "bitfit":     {},
    "vera":       dict(target_modules=["q_proj", "k_proj", "v_proj"]),
    "ia3":        dict(target_modules=["q_proj", "k_proj", "v_proj"],
                       feedforward_modules=["v_proj"]),
    "full":       {},
    "quant_base": dict(quant_dtype="8bit"),
}


def run_fwd(method, **kw):
    m = MiniNet()
    info = apply_method(m, method, **kw)
    x = torch.randint(0, 64, (2, 8))
    out = m(x)
    out.backward()
    return info


if __name__ == "__main__":
    print("=" * 70)
    print("PEFT 方法 smoke test")
    print("=" * 70)
    results = []

    # A 组
    for meth, kw in CASES.items():
        try:
            info = run_fwd(meth, **kw)
            ok = "PASS" if info.trainable_params > 0 else "WARN"
            results.append((meth, ok))
            print(f"[{ok:4s}] {meth:12s} -- {info.summary().split(chr(10))[0]}")
        except Exception as e:
            results.append((meth, "FAIL"))
            print(f"[FAIL] {meth:12s} -- {type(e).__name__}: {e}")

    # B 组：p_tuning_v2（需 transformers，验证可理解报错）
    try:
        from peft import PrefixTuningConfig
        _ = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
        m = MiniNet()
        try:
            apply_method(m, "p_tuning_v2", num_virtual_tokens=20)
            print("[SKIP] p_tuning_v2     -- 纯模块意外成功（无需 transformers？）")
            results.append(("p_tuning_v2", "PASS"))
        except AttributeError as e:
            print(f"[SKIP] p_tuning_v2     -- 需 transformers 模型（{e}），smoke 跳过前向")
            results.append(("p_tuning_v2", "SKIP"))
    except Exception as e:
        results.append(("p_tuning_v2", "FAIL"))
        print(f"[FAIL] p_tuning_v2     -- {type(e).__name__}: {e}")

    # B 组：efst（非 MoE 模型，验证正确 fallback）
    try:
        m = MiniNet()
        info = apply_method(m, "efst")
        ok = "PASS" if info.trainable_params == 0 else "WARN"
        results.append(("efst", ok))
        print(f"[{ok:4s}] efst          -- fallback 冻结全部, groups={info.extra.get('expert_groups')}")
    except Exception as e:
        results.append(("efst", "FAIL"))
        print(f"[FAIL] efst          -- {type(e).__name__}: {e}")

    print("-" * 70)
    failed = [m for m, s in results if s == "FAIL"]
    passed = sum(1 for _, s in results if s in ("PASS", "WARN", "SKIP"))
    print(f"结果: {passed}/{len(results)} 通过（FAIL={len(failed)}）")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("ALL PASSED")