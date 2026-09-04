# bench_r5_model.py - R5-4500U 真实模型 5 步训练对比
#   A: fp32 基座 + LoRA + adafactor        （基线）
#   B: fp32 基座 + LoRA + bnb.AdamW8bit    （8bit 优化器省内存）
#   C: 8bit 量化基座 + LoRA + AdamW8bit    （量化冻结层省内存）
# 输出：每步耗时 + 方案末 RSS + 进程峰值 RSS
# 运行：py -3.11 bench_r5_model.py
import ctypes
import ctypes.wintypes as wt
import gc
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(6)
torch.manual_seed(42)

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from transformers.optimization import Adafactor  # noqa: E402
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402

ROOT = os.path.dirname(HERE)  # deepsleep 根
BASE_MODEL = os.path.join(ROOT, "qwen3-1.7B")
DATA_FILE = os.path.join(ROOT, "jiaoben", "windows_code_clean.jsonl")


class _PMC(ctypes.Structure):
    """完整 PROCESS_MEMORY_COUNTERS（缺字段会让 GetProcessMemoryInfo 失败）。"""
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def rss_peak_mb():
    pm = _PMC(cb=ctypes.sizeof(_PMC))
    psapi = ctypes.windll.psapi
    # 必须显式声明签名：默认 32 位 c_int 会把 64 位 HANDLE/指针截断导致调用失败
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    h = ctypes.windll.kernel32.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(h, ctypes.byref(pm), pm.cb):
        return pm.WorkingSetSize / (1 << 20), pm.PeakWorkingSetSize / (1 << 20)
    return -1.0, -1.0


tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


class JsonlDataset(torch.utils.data.Dataset):
    """直接读 jsonl（instruction/output），不走 datasets 库（Windows filelock 坑）。"""

    def __init__(self, path, tok, max_len=64, n=5):
        import json
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if len(self.items) >= n:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                i, o = obj.get("instruction", ""), obj.get("output", "")
                if not i or not o:
                    continue
                msgs = [{"role": "system", "content": "You are a coding assistant."},
                        {"role": "user", "content": i},
                        {"role": "assistant", "content": o}]
                enc = tok(tok.apply_chat_template(msgs, tokenize=False),
                          truncation=True, max_length=max_len, padding=False)
                ids = torch.tensor(enc["input_ids"])
                self.items.append({"input_ids": ids, "labels": ids})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_loader():
    return torch.utils.data.DataLoader(JsonlDataset(DATA_FILE, tok), batch_size=1, shuffle=False)


def load_model():
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)
    m.config.use_cache = False
    return m


def run_steps(model, opt, steps=5, label=""):
    loader = make_loader()
    model.train()
    total = 0.0
    i = 0
    for batch in loader:
        if i >= steps:
            break
        t0 = time.perf_counter()
        out = model(input_ids=batch["input_ids"], labels=batch["labels"])
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        total += dt
        i += 1
        print(f"  [{label}] step {i}: loss {out.loss.item():.4f}  {dt:.2f}s", flush=True)
    rss, peak = rss_peak_mb()
    print(f"  [{label}] {i} 步训练耗时 {total:.2f}s ({total/i:.2f}s/步)  "
          f"RSS {rss:.0f}MB  进程峰值 {peak:.0f}MB", flush=True)
    return total, rss, peak


# ============ A: fp32 + LoRA + adafactor ============
print("=== A: fp32 + LoRA + adafactor (基线) ===", flush=True)
model = load_model()
model = get_peft_model(model, LoraConfig(
    r=2, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM))
opt = Adafactor([p for p in model.parameters() if p.requires_grad],
                scale_parameter=False, relative_step=False, lr=1e-4)
run_steps(model, opt, 5, "A")
del model, opt
gc.collect()

# ============ B: fp32 + AdamW8bit ============
print("=== B: fp32 + LoRA + AdamW8bit ===", flush=True)
import bitsandbytes as bnb  # noqa: E402
model = load_model()
model = get_peft_model(model, LoraConfig(
    r=2, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM))
opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-4)
run_steps(model, opt, 5, "B")
del model, opt
gc.collect()

# ============ C: 8bit 量化基座 + LoRA + AdamW8bit ============
print("=== C: 8bit 量化基座 + LoRA + AdamW8bit ===", flush=True)


class QuantLinearLora(nn.Module):
    """基座权重 8bit 量化存储 + 前向 dequantize + LoRA（fp32 可训练）。"""

    def __init__(self, weight: torch.Tensor, bias, lora_r=8, lora_alpha=16):
        super().__init__()
        from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise  # noqa: E402
        self.out_features, self.in_features = weight.shape
        w = weight.detach().float().reshape(-1)
        wq, stats = quantize_blockwise(w, blocksize=256)
        self.register_buffer("wq", wq)
        self.stats = stats
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().clone())
        else:
            self.register_buffer("bias", None)
        self.scaling = lora_alpha / lora_r
        self.lora_A = nn.Parameter(torch.randn(lora_r, self.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, lora_r))
        self.frozen_mem = self.wq.numel() + stats.absmax.numel() * 4
        self.orig_mem = weight.numel() * 4

    def forward(self, x):
        from bitsandbytes.functional import dequantize_blockwise  # noqa: E402
        w = dequantize_blockwise(self.wq, self.stats).reshape(self.out_features, self.in_features)
        out = F.linear(x, w, self.bias)
        out = out + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return out


def quantize_model_linear_lora(model, lora_r=8, lora_alpha=16):
    n = 0
    saved_mb = 0
    for name, m in list(model.named_children()):
        if isinstance(m, nn.Linear):
            q = QuantLinearLora(m.weight, m.bias, lora_r=lora_r, lora_alpha=lora_alpha)
            saved_mb += (q.orig_mem - q.frozen_mem) / (1 << 20)
            setattr(model, name, q)
            n += 1
        else:
            sn, sm = quantize_model_linear_lora(m, lora_r, lora_alpha)
            n += sn
            saved_mb += sm
    return n, saved_mb


model = load_model()
n, saved = quantize_model_linear_lora(model, lora_r=8, lora_alpha=16)
print(f"  量化替换 {n} 个 Linear，基座权重内存省约 {saved:.0f} MB", flush=True)
opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-4)
run_steps(model, opt, 5, "C")

print("\n=== 完成 ===", flush=True)
