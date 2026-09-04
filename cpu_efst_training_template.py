"""CPU MoE + Gated Delta Network + EFST 训练模板。

用法示例（在你有 torch/transformers/peft/bitsandbytes 的机器上）:

    python cpu_efst_training_template.py ^
        --base_model C:\\models\\some-moe ^
        --data_path C:\\data\\train.jsonl ^
        --cache_dir C:\\cache\\tokenized ^
        --output_dir C:\\out\\efst_model

    :: 启用硬盘均衡负载（自动模式，保护硬盘）:
    python cpu_efst_training_template.py ^
        --base_model C:\\models\\some-moe ^
        --data_path C:\\data\\train.jsonl ^
        --flash

    :: 速度优先（不管硬盘死活）:
    python cpu_efst_training_template.py ... --flash --flash_speed

    :: 手动指定每盘缓存 + 保留缓存文件:
    python cpu_efst_training_template.py ... --flash manual --flash_sizes C:2048,D:4096 --flash_keep

这个模板不是完整 Trainer，而是把下面几件事串起来：
1. torch_cpu_kit 设置 CPU 线程/内存策略；
2. bitsandbytes GDN patch（如果模型里有 Gated DeltaNet）；
3. EFST 只微调选中的专家（+ 可选 LoRA）；
4. 8-bit 优化器只给可训练参数分配状态；
5. DiskCachedDataset 做硬盘缓存，降低内存；
6. disk_balancer 硬盘均衡负载（可选，通过 --flash 启用）。
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# 方便直接从仓库根目录运行：能找到 cpu_bundle/pytorch 下的 efst.py / torch_cpu_kit.py
for _p in (os.path.join(_HERE, "cpu_bundle", "pytorch"), os.path.join(_HERE, "pytorch")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)


def parse_args():
    p = argparse.ArgumentParser(description="CPU MoE EFST training template")
    p.add_argument("--base_model", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--cache_dir", default="./cache_efst")
    p.add_argument("--output_dir", default="./out_efst")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--calib_batches", type=int, default=8)
    p.add_argument("--lora", action="store_true", help="给选中专家注入 LoRA")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--tune_router", action="store_true")

    # 硬盘均衡负载参数
    p.add_argument(
        "--flash", nargs="?", const="auto", default=None,
        help="启用硬盘均衡负载：auto/a=自动模式, manual=手动模式, true=启用自动模式",
    )
    p.add_argument(
        "--flash_sizes", type=str, default=None,
        help="手动模式每盘缓存大小(MB)，格式: C:2048,D:4096,E:8192",
    )
    p.add_argument(
        "--flash_paths", type=str, default=None,
        help="缓存路径，逗号分隔: D:\\cache,E:\\cache",
    )
    p.add_argument(
        "--flash_speed", action="store_true", default=False,
        help="速度优先模式（不管硬盘死活）",
    )
    p.add_argument(
        "--flash_keep", action="store_true", default=False,
        help="训练结束后保留缓存文件",
    )

    return p.parse_args()


def _parse_flash_config(args):
    """解析 --flash 相关参数，返回 DiskBalancerConfig 或 None。"""
    flash_val = args.flash
    if flash_val is None:
        return None

    from disk_balancer import DiskBalancerConfig

    raw_mode = str(flash_val).lower()
    if raw_mode in ("true", "1", "yes", "auto", "a"):
        mode = "auto"
    elif raw_mode in ("manual", "m"):
        mode = "manual"
    else:
        mode = "auto"

    sizes = {}
    if args.flash_sizes:
        for part in args.flash_sizes.split(","):
            mount, size = part.split(":")
            sizes[mount.strip().upper()] = int(size.strip())

    paths = []
    if args.flash_paths:
        paths = [p.strip() for p in args.flash_paths.split(",") if p.strip()]

    return DiskBalancerConfig(
        mode=mode,
        sizes=sizes,
        paths=paths,
        speed_first=args.flash_speed,
        keep_cache=args.flash_keep,
    )


def main():
    args = parse_args()

    # 0) 硬盘均衡负载（在 import torch 之前初始化配置，避免被 torch 抢占磁盘）
    flash_cfg = _parse_flash_config(args)
    balancer = None
    if flash_cfg is not None:
        from disk_balancer import DiskLoadBalancer
        balancer = DiskLoadBalancer(flash_cfg)

    # 1) CPU 环境：必须在 import torch 之前
    import torch_cpu_kit as tck
    tck.apply_env(verbose=True)

    import torch  # noqa: F401  (确保 import 顺序)
    tck.init(verbose=True)

    # 2) bitsandbytes GDN patch（Qwen3-Next / Qwen3.5 这类模型）
    try:
        import bitsandbytes as bnb
        from bitsandbytes.gdn_cpu import patch_transformers
        patch_transformers()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] bitsandbytes not available or patch failed: {e}")
        bnb = None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.train()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 0b) disk_balancer: 注册模型 + 启动（模型加载后）
    if balancer is not None:
        balancer.attach_model(model)
        balancer.start(script_dir=os.path.dirname(os.path.abspath(__file__)))

    # 3) EFST：专家专项微调
    from efst import EFSTConfig, apply_efst

    # 这里只是示意：实际数据加载请替换成你自己的 DataLoader
    dummy_loader = []

    efst_config = EFSTConfig(
        top_k=args.top_k,
        calibration_dataloader=dummy_loader,
        calibration_forward_fn=lambda batch: model(**batch) if isinstance(batch, dict) else model(batch),
        num_calibration_batches=args.calib_batches,
        tune_router=args.tune_router,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    info = apply_efst(model, efst_config)
    print(info.summary())

    # 4) 8-bit 优化器：只给可训练参数分配状态
    if bnb is not None:
        opt = bnb.optim.AdamW8bit(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )
    else:
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )

    # 5) 训练循环（示意；真实数据请用 DataLoader + DiskCachedDataset）
    os.makedirs(args.output_dir, exist_ok=True)
    step = 0
    try:
        for epoch in range(args.epochs):
            for batch in dummy_loader:
                # balancer: 检查内存，自动卸载冷参数
                if balancer is not None:
                    balancer.update_step()

                opt.zero_grad(set_to_none=True)
                out = model(**batch) if isinstance(batch, dict) else model(batch)
                loss = out.loss if hasattr(out, "loss") else out[0].mean()
                loss = loss / args.grad_accum
                loss.backward()
                if (step + 1) % args.grad_accum == 0:
                    opt.step()

                # 每 10 步：内存报告 + 磁盘缓存统计
                if step % 10 == 0:
                    rss, avail = tck.mem_report()
                    parts = [f"step {step} loss {loss.item():.4f} RSS {rss:.0f}MB avail {avail:.0f}MB"]
                    if balancer is not None:
                        s = balancer.stats()
                        parts.append(f"flash migrated={s['migrated_count']} cold_keys={s['cold_keys']} "
                                     f"mem={s['memory_percent']:.0f}%")
                    print("  ".join(parts))

                step += 1

    finally:
        # 训练结束：等待异步写盘 + 保存模型 + 清理磁盘缓存
        if balancer is not None:
            balancer.wait_writes()
        model.save_pretrained(args.output_dir)
        tok.save_pretrained(args.output_dir)
        if balancer is not None:
            balancer.cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
