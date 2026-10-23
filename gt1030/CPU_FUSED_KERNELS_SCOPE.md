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
  - silu [2048,11008] torch 31.28ms → native **18.53ms（1.71×）**；AVX2 多项式 exp，rel 4.7e-8，达标。

## 4. 未完成/后续

- **反向/autograd**：native 目前 forward-only；torch 参考可 autograd。训练接入需给 rms/silu/rope 写 backward（或自定义 Function）。
- **fused_ce 分块版**：`linear_cross_entropy(hidden, weight, labels, chunk)` 已实现（online logsumexp，不物化全 logits）；
  对拍 rel **1.2e-7**；M=1024/V=151936/D=2048 峰值 RSS **1.52GB vs full 2.68GB（0.57×）**，
  时间 3.38s vs 3.24s（约1.04×，基本持平）。native CE 未单独写。
- **AR 30M step 对比：不适用**（AR Hgemm 的 RMS/SiLU 在 device/OpenCL 上，host fused 不能替 device buffer）；
  要测 CPU 训练路径需给 `train_llm_sft.py`/CPU trainer 接 fused 模块（未做）。
- **rope native**：仅 torch 参考；未写 kernel。
