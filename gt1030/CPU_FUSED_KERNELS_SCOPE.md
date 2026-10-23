# CPU 融合算子包（Liger 式）范围与验收（2026-10-23）

## 1. 可复用清单

| 资产 | 位置 | 复用判断 |
|---|---|---|
| AR V2 RMSNorm stats/apply/bwd | `gt1030/full_gpu_layer.py` | 公式可复用；**OpenCL kernel，绑 GPU 布局**，不能直接作为 CPU 包 |
| AR V2 SiLU gate/bwd | 同上 | 同上 |
| AR V2 residual add / split / merge | 同上 | 同上 |
| CPU fused RMSNorm | `bitsandbytes/csrc/cpu_ops.cpp` (`fused_rmsnorm_fwd_cpu` 等) | **可直接复用为 native 实现**（已有 C 内核） |
| NF4 fused dequant+GEMM | `bnb.functional.gemv_4bit` | 已交付，推理默认走它（9–19× 物化） |
| 4-bit quant elementwise | `functional.py` | 可复用思路，非通用 fused API |
| CE | host `torch.nn.functional.cross_entropy` | 当前无 fused 实现；本期仅 torch 参考 |

## 2. 最小包 `bitsandbytes.fused`（已建）

```python
from bitsandbytes import fused
y = fused.rms_norm(x, w, eps=1e-6)       # torch reference（autograd OK）
y = fused.silu_mul(gate, up)             # torch reference
y = fused.rope(x, cos, sin)              # torch reference
loss = fused.fused_ce(logits, labels)    # torch reference
fused.native_available()                 # zig AVX2 DLL
y = fused.rms_norm_native(x, w)          # forward-only AVX2, 2.3–2.5×
y = fused.silu_mul_native(gate, up)      # forward-only C scalar（慢，实验性，不推荐）
```

- 实现：`bitsandbytes/fused/_ref.py`（契约 torch）、`_native.py`（ctypes）、`bnb_fused.cpp`（zig 编译 `bnb_fused.dll`）。
- 编译（本机 zig）：
  ```powershell
  & 'D:\work\tools-zig\zig\zig-x86_64-windows-0.14.1\zig.exe' c++ -O3 -mavx2 -mfma -shared `
    bitsandbytes\fused\bnb_fused.cpp -o bitsandbytes\fused\bnb_fused.dll
  ```

## 3. 单测/验收（已跑）

`gt1030/test_fused_ops.py`：native_available=True，ALL_PASS。
- rms_norm rel：4.49e-8 / 9.34e-8 / 7.03e-8（<1e-3）
- silu_mul rel：8.39e-9
- rope shape/finite；fused_ce 正常。
- 微基准（CPU 6线程）：
  - rms [2048,4096] torch 16.51ms → native **7.09ms（2.33×）**
  - rms [512,8192] 8.36→3.38ms（2.47×）
  - silu [2048,11008] torch 31.28ms → native **295.1ms（0.11×，慢）**；标量 exp 未向量化，**不推荐用 native silu**，等 AVX2 多项式 exp。

## 4. 未完成/后续

- **反向/autograd**：native 目前 forward-only；torch 参考可 autograd。训练接入需给 rms/silu/rope 写 backward（或自定义 Function）。
- **fused_ce native**：未实现；当前 torch。
- **AR 30M step 对比**：未接入；AR Hgemm 训练用 OpenCL V2 kernels，CPU fused 包不能直接替换其 device buffer；
  下一步需在 CPU SGEMM 路径或给 `full_gpu_layer.py` 加“host fused fallback”，再测 5–10 步 step/RSS。
- **rope native**：仅 torch 参考；未写 kernel。
