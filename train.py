# -*- coding: utf-8 -*-
"""train.py — 爆改 bnb 统一训练入口（CPU / AVX2，面向无显卡机器）。

一条命令切换 9 种训练方法：

    python train.py --method lora        --base_model <path> --data_path <jsonl>
    python train.py --method qlora       --base_model <path> --quant_dtype nf4
    python train.py --method p_tuning_v2 --base_model <path> --num_virtual_tokens 20
    python train.py --method bitfit      --base_model <path>
    python train.py --method vera        --base_model <path> --r 256
    python train.py --method ia3         --base_model <path>
    python train.py --method full        --base_model <path>
    python train.py --method quant_base  --base_model <path>   # 量化基座直接训练（真量化 + LSQ）
    python train.py --method efst        --base_model <path>   # MoE 专家微调
    python train.py --method rest        --base_model <path>   # 自我训练增强（RL，data_path=逐行 prompt）
    python train.py --method rloo        --base_model <path>   # 留一法强化学习（RL，data_path=逐行 prompt）

rest / rloo 需要额外的奖励函数参数（默认规则奖励）：
    --reward_pos good,helpful   # 命中加分的正关键词（逗号分隔）
    --reward_neg bad,toxic       # 命中减分的负关键词（逗号分隔）
    --base_peft lora             # RL 时注入的 PEFT 方法（默认 lora，冻结基座省内存）

每步都自动接入：
  - disk_balancer 硬盘均衡负载（--flash 系列参数）
  - torch_cpu_kit 线程 / 内存策略
  - 8bit 优化器（bitsandbytes CPU）
  - gpu_scheduler 核显(DirectML)块级常驻执行器（--igpu，实验性，默认关闭）

已知结论（技术报告 §3.7 / §7）：quant_base 已从旧 QAT（fp32 master 驻留、
不减内存）升级为真量化存储 + LSQ 可学习标度：基座权重真正存成 8bit/NF4/FP4
码字，fp32 master 不再驻留，8bit 省约 75% 权重内存、4bit 省约 87.5%；仅学
per-block 标度，基座码字不更新（需同时更新基座与低秩增量时改用 qlora）。
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "cpu_bundle", "pytorch"), os.path.join(_HERE, "pytorch")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# AVX 接口位
# ---------------------------------------------------------------------------
AVX_CHOICES = ("auto", "avx2", "avx512", "sse42")


def resolve_avx(choice: str) -> str:
    """解析 AVX 选择，返回 dispatch 目标（接口位）。

    当前只做占位：AVX2 是唯一实测路径，AVX-512 仅预留（未实现手写内核）。
    未来在这里接入 AVX-512 融合 quant/dequant 内核的分发。
    """
    c = choice.lower()
    if c == "auto":
        return "avx2"  # 当前机器均为 AVX2，无 AVX-512
    return c


# 未来 AVX-512 内核注册点（占位）
_AVX_KERNEL_REGISTRY = {}  # {target: callable}


def register_avx_kernels(target: str) -> None:
    """未来在 csrc 编译出 AVX-512 GDN/quant 融合内核后，在这里 dispatch。"""
    if target == "avx512":
        # TODO(AVX-512): import csrc avx512 kernels 并注册
        pass


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="爆改 bnb 统一训练入口")
    # 通用
    p.add_argument("--base_model", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="./out_train")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    # 方法
    p.add_argument("--method", required=True,
                   choices=["lora", "qlora", "p_tuning_v2", "bitfit", "vera",
                            "ia3", "full", "quant_base", "efst", "rest", "rloo"])
    # PEFT 参数
    p.add_argument("--target_modules", type=str, default=None,
                   help="逗号分隔，如 q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_virtual_tokens", type=int, default=20)
    p.add_argument("--quant_dtype", default="nf4", choices=["nf4", "fp4", "8bit"])
    p.add_argument("--cache_dequant", action="store_true", default=True)
    p.add_argument("--blocksize", type=int, default=None)
    # RL（rest / rloo）参数
    p.add_argument("--base_peft", default="lora",
                   help="RL 时注入的 PEFT 方法（默认 lora，冻结基座）")
    p.add_argument("--reward_pos", type=str, default=None,
                   help="正关键词，逗号分隔（命中加分）")
    p.add_argument("--reward_neg", type=str, default=None,
                   help="负关键词，逗号分隔（命中减分）")
    p.add_argument("--reward_length_bonus", type=float, default=0.0,
                   help="按生成长度加奖励（可选）")
    p.add_argument("--num_samples", type=int, default=4, help="ReST 每 prompt 采样条数")
    p.add_argument("--keep_ratio", type=float, default=0.5, help="ReST 保留 reward 最高比例")
    p.add_argument("--num_generations", type=int, default=2, help="RLOO 每 prompt 生成条数（>=2）")
    p.add_argument("--temperature", type=float, default=0.8, help="RL 采样温度")
    # AVX 接口位
    p.add_argument("--avx", default="auto", choices=AVX_CHOICES)
    # disk_balancer
    p.add_argument("--flash", nargs="?", const="auto", default=None,
                   help="启用硬盘均衡负载：auto/a=自动, manual=手动")
    p.add_argument("--flash_sizes", type=str, default=None, help="C:2048,D:4096(MB)")
    p.add_argument("--flash_paths", type=str, default=None, help="逗号分隔路径")
    p.add_argument("--flash_speed", action="store_true", help="速度优先")
    p.add_argument("--flash_keep", action="store_true", help="保留缓存")
    p.add_argument("--flash_threshold", type=float, default=0.8)
    # 核显(DirectML)块级常驻调度（实验性，默认关闭；详见 docs_cpu/TECH_REPORT.md §3.7）
    p.add_argument("--igpu", action="store_true",
                   help="启用核显(DirectML)块级常驻执行器（实验性；仅 batch*seq 短、"
                        "fp32 基座较小的场景有正收益，其余自动回退纯 CPU）")
    p.add_argument("--igpu_max_tokens", type=int, default=None,
                   help="核显执行器每步允许的最大 token 数（默认 256，或 GPU_SCHED_MAX_TOKENS）")
    p.add_argument("--igpu_mem_mb", type=int, default=None,
                   help="核显执行器允许的最大 fp32 基座内存 MB（默认按整机内存自适应："
                        "16GB→7000/12GB→5250，或 GPU_SCHED_MEM_MB）")
    p.add_argument("--igpu_min_gain", type=float, default=None,
                   help="核显启用门槛：启动时 GEMM 校准核显/CPU 吞吐比低于该值自动回退"
                        "纯 CPU（默认 1.10，或 GPU_SCHED_CALIB_MIN；0 跳过校准）")
    return p.parse_args()


def _parse_target_modules(s: str | None):
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _build_balancer(args):
    if args.flash is None:
        return None
    from disk_balancer import DiskBalancerConfig, DiskLoadBalancer
    raw = str(args.flash).lower()
    mode = "manual" if raw in ("manual", "m") else "auto"
    sizes = {}
    if args.flash_sizes:
        for part in args.flash_sizes.split(","):
            mount, size = part.split(":")
            sizes[mount.strip().upper()] = int(size.strip())
    paths = [x.strip() for x in args.flash_paths.split(",")] if args.flash_paths else []
    cfg = DiskBalancerConfig(
        mode=mode, sizes=sizes, paths=paths,
        speed_first=args.flash_speed, keep_cache=args.flash_keep,
        memory_threshold=args.flash_threshold,
    )
    return DiskLoadBalancer(cfg)


def _make_optimizer(model, lr: float):
    """优先用 bitsandbytes CPU 8bit 优化器，失败则退化为 AdamW。"""
    try:
        import bitsandbytes as bnb
        if hasattr(bnb.optim, "AdamW8bit"):
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    except Exception:
        pass
    return __import__("torch").optim.AdamW(model.parameters(), lr=lr)


def _load_prompts(data_path: str):
    """从纯文本文件逐行读取 prompt（RL 的 data_path 应为逐行 prompt，非 jsonl）。"""
    prompts = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def _run_rl(args, model, tokenizer):
    """rest / rloo 训练分支：冻结基座 + LoRA → 规则奖励 → RL。

    说明：RL 分支暂不接 disk_balancer（TRL 内部循环与 ReST 采样循环的
    内存策略由 CPU 卸载 / LoRA 冻结基座承担）。
    """
    from align_rl import (keyword_reward_fn, prepare_prompt_dataset,
                          rest_train, rloo_train)

    # 1) 注入 PEFT（默认 LoRA）冻结基座，只训 adapter，省内存
    from peft_backends import apply_method
    info = apply_method(
        model, args.base_peft,
        target_modules=_parse_target_modules(args.target_modules),
        r=args.r, alpha=args.alpha, dropout=args.lora_dropout,
    )
    print(info.summary())

    # 2) prompt + 规则奖励
    prompts = _load_prompts(args.data_path)
    pos = [w.strip() for w in args.reward_pos.split(",")] if args.reward_pos else []
    neg = [w.strip() for w in args.reward_neg.split(",")] if args.reward_neg else []
    if not pos and not neg and args.reward_length_bonus == 0.0:
        print("[rl] 警告：未提供 --reward_pos/--reward_neg，规则奖励恒为 0"
              "（所有样本同分，ReST 筛选退化、RLOO 无梯度信号）")
    reward_fn = keyword_reward_fn(pos_words=pos, neg_words=neg,
                                  length_bonus=args.reward_length_bonus)

    if args.method == "rest":
        res = rest_train(model, tokenizer, prompts, reward_fn,
                         num_samples=args.num_samples, keep_ratio=args.keep_ratio,
                         epochs=args.epochs, lr=args.lr,
                         max_new_tokens=args.max_length, temperature=args.temperature)
        print(f"[rest] {res}")
        model.save_pretrained(args.output_dir)
    else:  # rloo
        ds = prepare_prompt_dataset(prompts)
        trainer = rloo_train(model, tokenizer, ds, reward_fn,
                             num_train_epochs=args.epochs, lr=args.lr,
                             batch_size=args.batch_size,
                             num_generations=args.num_generations,
                             max_completion_length=args.max_length,
                             output_dir=args.output_dir)
        trainer.train()
        trainer.save_model(args.output_dir)
    print(f"[{args.method}] 完成，模型已保存到 {args.output_dir}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # 1) AVX 接口位
    avx_target = resolve_avx(args.avx)
    register_avx_kernels(avx_target)
    print(f"[avx] 选择目标: {avx_target}（AVX-512 为接口占位）")

    # 2) CPU 线程/内存策略
    try:
        from torch_cpu_kit import setup_optimal_threads
        setup_optimal_threads()
        print("[cpu_kit] 线程/内存策略已设置")
    except Exception as e:
        print(f"[cpu_kit] 跳过: {e}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 3) 加载模型 + tokenizer
    print(f"[model] 加载 {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3.5) 可选：核显(DirectML)块级常驻执行器（实验性，默认关闭）
    executor = None
    if args.igpu:
        if args.flash is not None:
            print("[igpu] 与 --flash 不兼容（内存策略不同），本次忽略 --igpu，走纯 CPU + flash")
        elif args.method in ("rest", "rloo"):
            print("[igpu] RL 分支暂不支持核显执行器，本次走纯 CPU")
        else:
            try:
                from gpu_scheduler import IgpuExecutor
                executor = IgpuExecutor(
                    model, max_tokens=args.igpu_max_tokens, max_mem_mb=args.igpu_mem_mb,
                    calib_min=(args.igpu_min_gain if args.igpu_min_gain else None),
                    calibrate=(False if args.igpu_min_gain == 0 else None))
                rs = executor.prepare(tokens=args.batch_size * args.max_length)
                if executor.ok:
                    print(f"[igpu] 已启用块级核显执行器：{executor.description()}")
                else:
                    print(f"[igpu] 未启用（{rs}），回退纯 CPU")
                    executor = None
            except Exception as e:
                print(f"[igpu] 核显执行器初始化失败，回退纯 CPU: {e}")
                executor = None

    # 3.6) RL 分支（rest / rloo）
    if args.method in ("rest", "rloo"):
        _run_rl(args, model, tokenizer)
        return

    # 4) 应用 PEFT 方法
    from peft_backends import apply_method
    info = apply_method(
        model, args.method,
        target_modules=_parse_target_modules(args.target_modules),
        r=args.r, alpha=args.alpha, dropout=args.lora_dropout,
        num_virtual_tokens=args.num_virtual_tokens,
        quant_dtype=args.quant_dtype, cache_dequant=args.cache_dequant,
        blocksize=args.blocksize,
    )
    print(info.summary())

    # 5) 优化器（核显执行器模式下使用 CPU 镜像训练，兼容 bnb 8bit 优化器）
    opt = _make_optimizer(executor.cpu_mirror if executor is not None else model, args.lr)

    # 6) disk_balancer
    balancer = _build_balancer(args)
    if balancer is not None:
        balancer.attach_model(model)
        balancer.start(script_dir=_HERE)
        print(f"[disk_balancer] 已启动: {balancer.stats().get('mode', '')}")

    # 7) 训练循环（示意；真实数据请接 DataLoader）
    os.makedirs(args.output_dir, exist_ok=True)
    step = 0
    print(f"[train] 开始 {args.epochs} epoch ...")
    try:
        for epoch in range(args.epochs):
            # 占位：这里替换为真实 DataLoader 迭代
            for _ in range(10):
                if balancer is not None:
                    balancer.update_step()
                ids = torch.randint(0, tokenizer.vocab_size,
                                    (args.batch_size, args.max_length))
                if executor is not None:
                    ids = ids.to(executor.dev)
                out = model(input_ids=ids, labels=ids)
                loss = out.loss / args.grad_accum
                loss.backward()
                if (step + 1) % args.grad_accum == 0:
                    if executor is not None:
                        executor.grad_to_cpu()      # 核显梯度 -> CPU 镜像
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        executor.weights_from_cpu() # 更新后的权重 -> 核显
                    else:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    gc.collect()
                print(f"  step {step} loss {loss.item() * args.grad_accum:.4f}")
    finally:
        if balancer is not None:
            balancer.cleanup()
    print("[train] 完成")


if __name__ == "__main__":
    main()