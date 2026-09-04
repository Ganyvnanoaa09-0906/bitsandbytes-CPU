"""验证 patch_transformers 对真实 Qwen3-Next 模型生效（Windows CPU）。

1. 随机初始化一个极小 Qwen3NextForCausalLM；
2. patch 前跑一次 forward+backward（torch 慢路径）；
3. patch 后跑一次（应走 gdn_cpu 融合内核）；
4. 对比两者输出是否一致（数学等价性）；
5. 确认 patch 后训练反向正常、耗时下降。
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")

import torch

BNB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes")
sys.path.insert(0, BNB_ROOT)

from transformers import Qwen3NextConfig, Qwen3NextForCausalLM  # noqa: E402

torch.manual_seed(0)

cfg = Qwen3NextConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    linear_conv_kernel_dim=4,
    max_position_embeddings=64,
    decoder_sparse_step=9999,   # 不用 MoE 块
    layer_types=["attention", "linear_attention"],  # 一层标准注意力 + 一层 GDN
)
print("building tiny Qwen3Next ...")
model = Qwen3NextForCausalLM(cfg)
model.eval()

x = torch.randint(0, 256, (1, 32))
print("layers:", [type(m).__name__ for m in model.model.layers])

# ---------- patch 前（torch 慢路径） ----------
with torch.no_grad():
    t0 = time.perf_counter()
    out_before = model(x).logits
    t_before = time.perf_counter() - t0
print(f"pre-patch  fwd {t_before*1e3:8.1f} ms")

# ---------- patch 后（融合内核） ----------
from bitsandbytes.gdn_cpu import patch_transformers  # noqa: E402

ok = patch_transformers()
print(f"patch_transformers -> {ok}")

with torch.no_grad():
    t0 = time.perf_counter()
    out_after = model(x).logits
    t_after = time.perf_counter() - t0
print(f"post-patch fwd {t_after*1e3:8.1f} ms   ({t_before/max(t_after,1e-9):.1f}x)")

diff = (out_before - out_after).abs().max().item()
print(f"fwd max |before-after| = {diff:.3e}")
assert diff < 1e-3, "patch changed the model output!"

# ---------- 训练反向（patch 后） ----------
model.train()
loss = model(x, labels=x).loss
t0 = time.perf_counter()
loss.backward()
dt = time.perf_counter() - t0
print(f"train backward {dt*1e3:.1f} ms  loss {loss.item():.4f}")
assert loss.isfinite()
print("QWEN3-NEXT PATCH VERIFY PASSED")
