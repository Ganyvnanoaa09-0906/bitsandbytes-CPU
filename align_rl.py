# -*- coding: utf-8 -*-
"""align_rl.py — 对齐 / 强化学习（RL）方法：ReST 与 RLOO（CPU 训练用）。

补全 DPO / KTO / CPO / SimPO / ORPO（已在 D:\\work\\使用速查.txt 实测）之后的
两条 RL 路线：

    ReST  Reinforced Self-Training（自我训练增强）
          采样 → reward 打分 → 筛选高分 → 在筛选数据上 SFT（cross-entropy）。
          不是真正的 policy gradient，但实现简单、CPU 也能跑。

    RLOO  REINFORCE Leave-One-Out（留一法强化学习）
          真正的 policy gradient，用组内 leave-one-out baseline 降方差。
          复用 TRL 的 RLOOTrainer，适配 CPU（use_cpu=True）。

两者都依赖一个 reward 函数（CPU 环境通常没有独立 reward model，
可用规则 / 关键词 / 参考模型 logprob 差做奖励）。本模块提供示例 reward_fn，
实际使用请替换成你的打分逻辑。

与现有基座配合：
  - 采样 / SFT 复用 PEFT 包裹的模型 + bnb 8bit 优化器（AdamW8bit 可用时优先）；
  - RLOO 的 RLOOTrainer 支持 peft_config 透传，与现有 LoRA 流程一致。
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Sequence

# 爆改 bnb 仓库根：确保 import 到 CPU 后端版本（有正确 __version__），
# 否则 transformers 的 is_bitsandbytes_available() 会因版本 'N/A' 崩溃。
_HERE = os.path.dirname(os.path.abspath(__file__))
_BNB_ROOT = os.path.join(_HERE, "bitsandbytes")
if os.path.isdir(os.path.join(_BNB_ROOT, "bitsandbytes")):
    sys.path.insert(0, _BNB_ROOT)

import torch

__all__ = [
    "keyword_reward_fn",
    "rest_train",
    "rloo_train",
    "prepare_prompt_dataset",
]


# ---------------------------------------------------------------------------
# 示例 reward 函数
# ---------------------------------------------------------------------------

def keyword_reward_fn(pos_words: Sequence[str] = (),
                      neg_words: Sequence[str] = (),
                      length_bonus: float = 0.0) -> Callable[[List[str], dict], List[float]]:
    """规则奖励：命中正关键词加分、负关键词减分、可选长度奖励。

    返回一个 ``reward_fn(completions, **kwargs) -> list[float]``，
    兼容 TRL RLOOTrainer 的 reward_funcs 签名。
    """
    def _reward(completions: List[str], **kwargs) -> List[float]:
        scores = []
        for c in completions:
            s = 0.0
            for w in pos_words:
                if w in c:
                    s += 1.0
            for w in neg_words:
                if w in c:
                    s -= 1.0
            s += length_bonus * len(c)
            scores.append(s)
        return scores
    return _reward


# ---------------------------------------------------------------------------
# 通用优化器（优先 bnb 8bit，退化 AdamW）
# ---------------------------------------------------------------------------

def _make_optimizer(model, lr: float):
    try:
        import bitsandbytes as bnb
        if hasattr(bnb.optim, "AdamW8bit"):
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    except Exception:
        pass
    return torch.optim.AdamW(model.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def prepare_prompt_dataset(prompts: Sequence[str]) -> "list":
    """把 prompt 文本列表转成 TRL RL 需要的 dataset（含 prompt 列）。"""
    try:
        from datasets import Dataset
        return Dataset.from_dict({"prompt": list(prompts)})
    except Exception:
        return [{"prompt": p} for p in prompts]


# ---------------------------------------------------------------------------
# ReST：采样 → 打分 → 筛选 → SFT
# ---------------------------------------------------------------------------

def _to_device(enc, model):
    """把 tokenizer 输出移到模型设备上，兼容 dict / BatchEncoding。"""
    device = next(model.parameters()).device
    if isinstance(enc, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
    return enc.to(device) if hasattr(enc, "to") else enc


def rest_train(model, tokenizer, prompts: Sequence[str],
               reward_fn: Callable[[List[str]], List[float]],
               num_samples: int = 4, keep_ratio: float = 0.5,
               epochs: int = 1, lr: float = 1e-5,
               max_new_tokens: int = 64, temperature: float = 0.8,
               verbose: bool = True) -> dict:
    """Reinforced Self-Training 一个迭代：采样→打分→筛选→SFT。

    Args:
        model: policy 模型（建议 PEFT 包裹，冻结基座只训 adapter）。
        tokenizer: 分词器（需有 pad_token 或 eos_token 用于 generate）。
        prompts: prompt 文本列表。
        reward_fn: callable(completions) -> list[float]。
        num_samples: 每个 prompt 采样条数。
        keep_ratio: 保留 reward 最高的比例（0~1）。
        epochs: SFT 轮数。
        lr: SFT 学习率。
        max_new_tokens / temperature: 采样参数。
        verbose: 是否打印进度。

    Returns:
        dict: {"sampled", "selected", "best_score", "mean_score"}。
    """
    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id

    # 1) 采样
    completions: List[str] = []
    model.eval()
    with torch.no_grad():
        for p in prompts:
            enc = _to_device(tokenizer(p, return_tensors="pt"), model)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_samples,
                pad_token_id=pad_id,
            )
            completions.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    # 2) 打分
    scores = reward_fn(completions)

    # 3) 筛选（保留 reward 最高）
    pairs = sorted(zip(completions, scores), key=lambda x: -x[1])
    keep_n = max(1, int(len(pairs) * keep_ratio))
    selected = [t for t, _ in pairs[:keep_n]]

    if verbose:
        print(f"[ReST] sampled={len(completions)}, selected={keep_n}, "
              f"best_score={pairs[0][1]:.3f}, "
              f"mean_score={sum(scores) / len(scores):.3f}")

    # 4) SFT（cross-entropy）
    opt = _make_optimizer(model, lr)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for t in selected:
            enc = _to_device(tokenizer(t, return_tensors="pt"), model)
            labels = enc["input_ids"].clone()
            out = model(**enc, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.detach())
        if verbose:
            print(f"[ReST] epoch {ep + 1}/{epochs} loss={total / max(1, len(selected)):.4f}")

    return {"sampled": len(completions), "selected": keep_n,
            "best_score": pairs[0][1], "mean_score": sum(scores) / len(scores)}


# ---------------------------------------------------------------------------
# RLOO：复用 TRL RLOOTrainer
# ---------------------------------------------------------------------------

def rloo_train(model, tokenizer, train_dataset,
               reward_fn: Callable[[List[str], dict], List[float]],
               num_train_epochs: int = 1, lr: float = 1e-5,
               batch_size: int = 1, num_generations: int = 2,
               max_completion_length: int = 256, output_dir: str = "./rloo_out",
               **extra):
    """构造 TRL RLOOTrainer（REINFORCE Leave-One-Out），适配 CPU。

    Args:
        model: policy 模型（可以是 PEFT 包裹，或裸模型 + extra["peft_config"]）。
        tokenizer: 分词器。
        train_dataset: 含 "prompt" 列的 dataset（见 prepare_prompt_dataset）。
        reward_fn: callable(completions, **kwargs) -> list[float]。
        其余同 TRL RLOOConfig。

    Returns:
        RLOOTrainer 实例（未 train，调用 trainer.train() 开始）。
    """
    from trl import RLOOTrainer, RLOOConfig

    # RLOO 要求 generation_batch_size（= per_device_train_batch_size × steps_per_generation）
    # 必须能被 num_generations 整除，否则分组打分时 prompt 组不完整。
    if batch_size % num_generations != 0:
        orig = batch_size
        batch_size = max(num_generations, (batch_size // num_generations + 1) * num_generations)
        print(f"[RLOO] per_device_train_batch_size {orig} 不能被 num_generations "
              f"{num_generations} 整除，自动调整为 {batch_size}")

    args = RLOOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        optim="adafactor",
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        # RLOO 用 leave-one-out baseline，无 KL 项，不需要参考模型；beta=0 省内存且跳过 ref_model 加载。
        beta=0.0,
        bf16=False,
        fp16=False,
        use_cpu=True,
        report_to="none",
        **{k: v for k, v in extra.items()
           if k in RLOOConfig.__dataclass_fields__},
    )

    trainer = RLOOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=extra.get("peft_config"),
    )
    return trainer