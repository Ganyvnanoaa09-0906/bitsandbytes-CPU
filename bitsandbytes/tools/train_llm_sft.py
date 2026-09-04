#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/train_llm_sft.py — 用本地 LLM + 真实 jsonl 文本做 LoRA SFT 训练（CPU / R5）。
   用途：验证爆改 bnb 在 LLM 真实文本训练上的功能正确性 + 记录速度/内存/loss。
   用法：
     py tools/train_llm_sft.py --model D:\\work\\textmodel\\qwen3.5-0.8B --data D:\\work\\sujvji\\erotiquant2\\erotiquant2.jsonl --steps 500
   说明：qwen3.5-0.8B 为 Qwen3_5ForCausalLM(混合注意力+MoE)，用 peft.LoraConfig 冻结基座只训
         LoRA adapter，配 bnb.optim.AdamW8bit，纯 CPU fp32。
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import bitsandbytes as bnb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="jsonl 或 json 数组文件，含 text 字段")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=r"D:\work\output\llm_sft")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dtype = torch.float32

    t = AutoTokenizer.from_pretrained(args.model)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    print(f"[load] {args.model} params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")

    # LoRA 冻结基座
    try:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.rank, lora_alpha=args.rank*2, lora_dropout=0.0,
                         task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj"])
        m = get_peft_model(m, cfg)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[lora] trainable={trainable/1e6:.2f}M  (base {sum(p.numel() for p in m.parameters())/1e6:.1f}M)")
    except Exception as e:
        print(f"[warn] peft LoRA fail ({e}); fallback full finetune (slow)")
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

    # 读 jsonl(数组或逐行)
    raw = open(args.data, encoding="utf-8").read()
    s = raw.lstrip()
    if s.startswith("["):
        items = json.loads(raw)
        texts = [x["text"] for x in items if x.get("text")]
    else:
        texts = [json.loads(l)["text"] for l in raw.splitlines() if l.strip()]
    print(f"[data] {len(texts)} 条文本")

    # 预处理 tokenize
    enc = t(texts[:200], truncation=True, max_length=args.seq, padding="max_length",
            return_tensors="pt")
    # 循环取 batch
    ids = enc["input_ids"]

    opt = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, m.parameters()), lr=args.lr)
    m.train()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    losses = []
    if args.steps < 1:
        raise ValueError(f"--steps 必须 >=1，得到 {args.steps}")
    if args.batch < 1:
        raise ValueError(f"--batch 必须 >=1，得到 {args.batch}")
    if not texts:
        raise ValueError(f"--data 无可用文本（{args.data} 没有含 text 的记录）")
    for step in range(args.steps):
        b = step % ids.shape[0]
        input_ids = ids[b:b+args.batch]
        labels = input_ids.clone()
        opt.zero_grad()
        out = m(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 10 == 0:
            el = time.time() - t0
            print(f"  step {step:4d} | loss {loss.item():.4f} | {el:.0f}s ({step/max(el,1):.2f} step/s)")

    t = time.time() - t0
    print("\n--- LLM SFT results ---")
    print(f"steps: {args.steps}  seq: {args.seq}  rank: {args.rank}")
    if not losses:
        print("WARNING: 0 steps -- no losses recorded (did you pass --steps >= 1?).")
        return
    print(f"loss first {losses[0]:.4f} -> last {losses[-1]:.4f} (avg {sum(losses)/len(losses):.4f})")
    print(f"total {t:.1f}s  {args.steps/t:.2f} step/s  peak RSS ~{torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 'n/a'}")
    print(f"OK: LLM SFT training complete")


if __name__ == "__main__":
    main()
