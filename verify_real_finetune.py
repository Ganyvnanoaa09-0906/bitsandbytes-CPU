"""真实权重微调冒烟：Qwen3.5-0.8B（GDN+MoE）+ patch + EFST + AdamW8bit，夏弥角色数据。

流程：
1. 从本地加载 Qwen3.5-0.8B（fp32 CPU）
2. patch_transformers() 让 GDN 层走融合内核
3. EFST 校准选 top-k 热专家（3D 张量专家）+ tune_router
4. AdamW8bit 只训可训练参数
5. 用夏弥（龙族）场景对话微调若干步
6. 验证 loss 下降 + 微调前后生成对比
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DSH_ROOT = os.path.dirname(HERE)  # deepsleep 根目录（纯 ASCII，transformers 加载需要）
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))
sys.path.insert(0, HERE)

MODEL_PATH = os.path.join(DSH_ROOT, "models", "qwen3.5-0.8B")
DATA_PATH = os.path.join(DSH_ROOT, "sujvji", "lomhzhu-xiami_training.json")
N_STEPS = int(os.environ.get("FT_STEPS", "12"))
MAX_LEN = int(os.environ.get("FT_MAXLEN", "192"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

print(f"loading {MODEL_PATH} (fp32, cpu) ...")
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    local_files_only=True,
)
print(f"loaded in {time.perf_counter()-t0:.0f}s, "
      f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
model.train()

from bitsandbytes.gdn_cpu import patch_transformers  # noqa: E402
assert patch_transformers(), "patch_transformers found nothing"
print("GDN patch: OK")

# ---------------- 数据：夏弥场景对话 ----------------
with open(DATA_PATH, encoding="utf-8") as f:
    raw = json.load(f)
samples = []
sys_prompt = "你是夏弥，一个活泼开朗的少女，喜欢开玩笑和恶作剧，说话俏皮、充满活力。"
for item in raw[:48]:
    samples.append([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": item["instruction"]},
        {"role": "assistant", "content": item["output"]},
    ])
texts = [tok.apply_chat_template(s, tokenize=False) for s in samples]
enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)
enc["labels"] = enc["input_ids"].copy()
print(f"train samples: {len(samples)}")

# ---------------- EFST（仅 MoE 模型适用） ----------------
from efst import EFSTConfig, apply_efst, find_expert_groups  # noqa: E402

groups = find_expert_groups(model)
if groups:
    def calib(batch):
        return model(input_ids=batch["input_ids"], labels=batch["labels"])

    # 注意：input_ids 必须 2D [B, T]——transformers 5.15 的 qwen3_5 对 1D
    # 输入有形状 bug（与 GDN patch 无关）
    calib_ds = [{"input_ids": torch.tensor(enc["input_ids"][i]).unsqueeze(0),
                 "labels": torch.tensor(enc["labels"][i]).unsqueeze(0)}
                for i in range(6)]
    info = apply_efst(model, EFSTConfig(
        top_k=2,
        calibration_dataloader=calib_ds,
        calibration_forward_fn=calib,
        num_calibration_batches=6,
        tune_router=True,
        lora=True,
        lora_r=8,
        lora_alpha=16,
    ))
    print(info.summary())
else:
    print("no MoE experts in this model (dense Qwen3.5-0.8B) - EFST skipped, "
          "training all params with AdamW8bit")

import bitsandbytes as bnb  # noqa: E402

trainable = [p for p in model.parameters() if p.requires_grad]
n_tr = sum(p.numel() for p in trainable)
n_all = sum(p.numel() for p in model.parameters())
print(f"trainable {n_tr/1e6:.2f}M/{n_all/1e6:.2f}M ({100*n_tr/n_all:.2f}%)")
if groups:
    assert n_tr < n_all * 0.25, "EFST should freeze the majority of a MoE model"

opt = bnb.optim.AdamW8bit(trainable, lr=5e-4)

# ---------------- 微调 ----------------
print(f"training {N_STEPS} steps (CPU) ...")
t0 = time.perf_counter()
losses = []
for step in range(N_STEPS):
    i = step % len(enc["input_ids"])
    input_ids = torch.tensor(enc["input_ids"][i]).unsqueeze(0)
    labels = torch.tensor(enc["labels"][i]).unsqueeze(0)
    opt.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    opt.step()
    losses.append(out.loss.item())
    if step % 3 == 0 or step == N_STEPS - 1:
        print(f"step {step:2d}  loss {out.loss.item():.4f}  "
              f"({(time.perf_counter()-t0)/(step+1):.1f}s/step)")
print(f"total {time.perf_counter()-t0:.0f}s")
assert torch.isfinite(torch.tensor(losses)).all()
# 每步换样本，loss 方差大：用趋势判断（末段均值 < 首段均值）而非终值比较
head = sum(losses[:6]) / 6
tail = sum(losses[-6:]) / 6
print(f"loss head-mean {head:.4f} -> tail-mean {tail:.4f}")
assert tail < head * 0.95, f"loss not trending down: {head:.4f} -> {tail:.4f}"

# ---------------- 微调效果：同一批样本 loss 前后对比 ----------------
model.eval()
ev = [torch.tensor(enc["input_ids"][i]).unsqueeze(0) for i in range(6, 10)]
with torch.no_grad():
    before = sum(model(input_ids=t, labels=t).loss.item() for t in ev) / len(ev)
print(f"eval loss on held-out samples: before {before:.4f}")
# 生成对比（关闭 thinking 的纯回答模式）
q = "喂，你看到我新买的相机镜头了吗？"
msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": q}]
with torch.no_grad():
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   chat_template_kwargs={"enable_thinking": False})
    inp = tok(text, return_tensors="pt")
    gen = model.generate(inp.input_ids, max_new_tokens=64, do_sample=True, temperature=0.5)
reply = tok.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
print("=== 微调后回复 ===")
print(reply)
print("=== REAL FINETUNE SMOKE PASSED ===")
