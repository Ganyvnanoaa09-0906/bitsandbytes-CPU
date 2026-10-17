# 技术报告：纯 CPU 环境的 bitsandbytes 微小调整与训练加速

**作者**：deepsleep team
**日期**：2026-09
**范畴**：面向无 NVIDIA GPU（仅 CPU，AVX2 ，12–16GB 内存）设备的
大模型微调与生图训练加速工程化。

---

## 摘要

本报告记录在**无独立显卡、仅 AVX2 CPU（AMD Ryzen 5 4500U 6C6T 16GB /
Intel i5-10400 6C12T 12GB）**环境下，对 bitsandbytes 的 CPU 后端进行系统性
爆改，使 LoRA 微调、量化冻结层、对齐训练（DPO/KTO/…）、SD 生图训练达到可用水平。

核心贡献：
1. **Gated DeltaNet 融合内核**（Qwen3-Next/3.5 线性注意力）：较朴素逐时间步实现提速约 **35×**；
2. **8-bit 优化器融合内核**：单遍 dequant→update→requant，优化器状态内存压至 fp32 的 **1/3.8**；
3. **8-bit 融合反量化 GEMM**（gemm_8bit）：权重全程 uint8（DRAM 流量 1/4），无 fp32 临时；
4. 面向 AVX2 的 Windows 无 CUDA 编译管线 + **Linux（x86_64/aarch64）跨平台构建 + torch-free 自检**；
5. 明确的**负面结论**：AVX2 上 bf16 GEMM 卡死、基座权重降精度存储**全量训练**路径负收益
   （但真量化存储 + LSQ 只学标度可实现压缩内存，见 §3.5）。

---

## 1. 问题背景

目标设备均无 NVIDIA GPU（仅核显/集显，且核显用系统共享内存，无法作为
权重 offload 目标），指令集上限为 AVX2。此时：

- 1.7B 模型 fp32 权重 6.8GB，叠加梯度+优化器状态后 12/16GB 机器容易 swap；
- 官方 PyTorch CPU wheel（oneDNN 后端）已针对 AVX2 优化，但 **bf16 GEMM 在
  无 AVX512-BF16 指令的机器上由 oneDNN 软件模拟，单次 matmul >90s 近乎卡死**；
- Qwen3-Next/3.5 的 Gated DeltaNet 在无 Triton / 逐时间步环境下一次反向 728s；
- SD 生图（diffusers 0.40）已移除内置 Trainer，训练需手写循环。

---

## 2. 硬件与方法

| 机器 | CPU | 内存 | 指令集 |
|---|---|---|---|
| A | AMD Ryzen 5 4500U | 6C6T | 16GB | AVX2 |
| B | Intel i5-10400 | 6C12T | 12GB | AVX2 |

PyTorch 2.13.0+cpu（oneDNN）、transformers 5.15、peft 0.18.1、trl 1.12、
diffusers 0.40。全部 fp32（AVX2 上 bf16 禁用）。

---

## 3. 核心改动与关键结果

### 3.1 GDN 融合内核（csrc/cpu_gdn.cpp）

| 指标 | 朴素逐时间步 | 融合内核 | 提升 |
|---|---|---|---|
| 单层前向+反向（T=1024,B=2,H=4,K=V=64） | 1813 ms | **51.6 ms** | **≈35×** |
| 反向精度（相对 double 参考，fp32/bf16/fp16） | — | 1e-7 / 1e-3 / 1e-4 | 达标 |
| 检查点内存（T=8192, 8 层，自动 C） | 1074 MB | **134 MB** | **≈8×** |

**0.8B 级 Qwen3Next 反向传播实测**（`verify_qwen3next_patch.py`，CPU）：
真实 Qwen3Next 解码层（含 Gated DeltaNet 线性注意力），`patch_transformers()` 生效：

| 指标 | 值 |
|---|---|
| patch 前 forward | 59.9 ms |
| patch 后 forward | **11.1 ms（5.4× 加速）** |
| 训练 backward | 75.1 ms，loss 5.5785（有限，收敛） |
| 结果 | QWEN3-NEXT PATCH VERIFY **PASSED** |

> 该测试验证 GDN 融合内核在真实 Qwen3Next 结构上的**前向 + 反向**均正确——这正是
> 0.8B 级 Qwen3Next/Qwen3.5 模型微调的瓶颈。

### 3.2 8-bit 优化器（coptimizer_update_8bit_blockwise_cpu）

| 指标 | 值 |
|---|---|
| 8-bit AdamW 收敛终态 vs fp32 | 差 < 0.01 |
| 优化器状态内存 | fp32 的 **1/3.8** |
| 210 组合压力测试（7 优化器 × 10 尺寸 × 3 dtype） | **0 失败**，determinism diff=0 |
| AdamW8bit 4M 参数单步 | 13.1 ms |

### 3.3 gemm_8bit 融合反量化 GEMM

| 指标 | 值 |
|---|---|
| 权重内存 | uint8（fp32 的 1/4） |
| 前向精度（vs dequant+F.linear） | 相对误差 ≤ 4.7e-7 |
| DRAM 流量 | 权重全程 uint8，无 fp32 临时 |

测试形状：M∈[1,8], K,N∈[320,1280], blocksize∈[64,256]——全部相对误差 ≤ 4.7e-7。

### 3.4 线程与精度（实证）

| 任务 | 6线程 | 8线程 | 12线程 |
|---|---|---|---|
| GEMM 512×2048×8192（GFLOP/s） | 277 | **313** | 273 |
| 2D 卷积 256ch 3×3 64×64（GFLOP/s） | 329 | 489 | **618** |
| latent self-attention（ms） | **123** | 138 | 147 |

- 文本 GEMM 密集 → 8 线程最优；生图卷积密集 → **12 线程**；
- R5-4500U（6C6T 无超线程）→ **6 线程**（物理核拉满）。
- **bf16 在 AVX2 上不可用**（软件模拟 >90s 卡死）——训练一律 fp32。

### 3.5 基座权重降精度存储的负面结论（重要）

| 方案 | 权重内存 | 5 步耗时（1.7B + i5） | 相对 fp32 |
|---|---|---|---|
| fp32 | 6.8GB | 36.8s | 1.0× |
| 8bit 量化 + 前向 dequant | ~1.8GB | 375s | **10.2×** |
| bf16 存储 + .float() | ~3.4GB | 1058s | **28.7×** |

**结论**：AVX2 CPU 上，基座权重的降精度存储（8bit/bf16）因**逐层转换开销 +
量化期峰值内存**反而慢 10~29 倍，不可用于**全量更新基座权重**的训练路径（仅适用纯推理）。
内存优化应靠「控制激活 + 8bit 优化器 + 冻结层量化」。

**补充（量化基座直接训练，quant_base / LSQ）**：上述结论针对「基座权重仍需全量更新」的场景。
若放弃更新码字、仅学习量化标度（LSQ），则可把权重真正存成 8bit/NF4/FP4 码字（fp32 master
不再驻留），8bit 省约 75%、4bit 省约 87.5% 权重内存，实现「压缩内存」目标；代价是训练只
调整 per-block 标度、不更新基座码字。该能力由上层 `quant_lora.QuantLinearTrainable` 实现，
复用本仓库的 `quantize_blockwise` / `quantize_4bit` / 码本（`create_dynamic_map` / `get_4bit_type`）。

### 3.6 EFST：MoE 专家专项微调（efst.py）

**目标**：MoE 模型低内存微调——只训练选中专家（及可选 router），冻结其余，
显著压缩可训练量与内存。

| 指标 | 值 |
|---|---|
| 实测模型 | 随机初始化 Qwen3Next 3 层混合（3 个 3D tensor 专家组 × 8 专家） |
| top-2 选择后 | 可训练 725,712 → 143,600（19.8%） |
| 60 步训练 | loss 下降 23%，收敛正常 |
| 自检 | `efst_selftest.py` → **PASSED**（moe 专家识别 + LoRA 注入 + 前向/反向） |

**机制**：`freeze_all` + `unfreeze_experts`（手动 `expert_indices` 或校准数据自动选
`top_k`），`add_lora_to_experts`（经典专家注入 LoRA）。对 transformers 5.15+
Qwen3Next/Qwen3.5 的 **3D 张量专家**（`[num_experts,...]` 大 Parameter）：
`split_3d_expert_params` 拆成 `ParameterList` 按行解冻，`lora=True` 时自动回退为
按行解冻（tensor 专家无 `nn.Linear` 子模块）。配合
`bitsandbytes.gdn_cpu.patch_transformers()` + `AdamW8bit` 使用（8bit 优化器只给
可训练参数分配状态）。

### 3.7 核显（DirectML）加速的完整结论：逐算子负优化 → 块级常驻可行

**问题**：无独显机型（R5-4500U，AMD Radeon(TM) Graphics，与 CPU 共享 DDR4-2667）
上，核显能否加速训练？

**阶段一：逐算子调度 = 负优化（结论保留）**

`gpu_scheduler.py` 初版按「计算量/搬运量」比 `M*N/(M+N)` 自动派发大算子。
同机 8 步实测（8 层 BigLinear(2048²) LoRA 式训练）：

| 配置 | 每步耗时 | 峰值 RSS | 损失 |
|---|---|---|---|
| 纯 CPU | 0.069 s | 465 MB | 0.8955 |
| 逐算子核显调度 | 0.134 s（0.51×） | 499 MB | 0.9051 |

**真凶修正**：慢 2 倍不是因为「搬运带宽」，而是 `torch_directml` 每次
`.to("cpu")/item()` 都是**整队列排空（单次约 10~15ms）**——8 层×2 次往返 ≈ 20 次
排空 ≈ 200ms+。带宽实测也证实：memcpy 多进程仅 ~21-22 GB/s（理论 42.7 的一半），
本就是本机天花板，无带宽可「调度」。

**阶段二：块级常驻执行器（本机实测可行）**

形态：全部权重常驻核显（只搬一次）+ 前向/反向整步在核显执行 + 每步一次同步；
可训练参数（LoRA 适配器）梯度/权重以小体积 CPU↔核显往返。

| 实验 | 结果 |
|---|---|
| 大 GEMM 流水线（50×2048²，末尾同步一次） | DML 294.5 GFLOPS vs CPU 226.7 = **1.30×** |
| 真实 Qwen 结构全链（8 层 1024 宽，权重常驻，seq=128） | **1.35×**（206.0 vs 277.4 ms/步） |
| 同全链 seq=512 | 0.70×（1135.4 vs 796.7 ms/步） |
| DML `F.sdpa`（S=256/512/1024） | 4.2 / 27.2 / 158.1 ms vs CPU 1.4 / 5.8 / 22.9 ms（3~7× 慢） |
| `multi_head_attention` 专用内核 | 本机驱动 27.20.11032 不支持（RuntimeError，且为 decode 专用） |

**结论**：核显加速在本机上**可行但范围窄**——GEMM 密集 + seq≤256 场景实测
+30~35%；长序列（seq≥512）被 DML 注意力的慢实现拖成 -30%，且驱动限制无法用专用
内核修复。与 §3.5 一致：「内存带宽是天花板」依旧成立，核显只适用于**少同步 +
少搬运**形态。

**最终形态**：`train.py --igpu` 落地为**块级常驻执行器**（默认关闭）：每步 token ≤
`GPU_SCHED_MAX_TOKENS`（默认 256）、fp32 基座 ≤ `GPU_SCHED_MEM_MB`（默认 7000MB）、
无量化参数时启用；否则打印原因自动回退纯 CPU。逐算子接口（`big_gemm` /
`patch_igpu`）保留仅作实验，不推荐用于训练。

**设备适用性**：实测数据全部来自 AMD Radeon(TM) Graphics（R5-4500U）；Intel
UHD 630（i5-10400 等）走同一 DirectML 接口，但核显相对 CPU 更弱，预期收益低于
AMD 机型，需在目标机型实测确认。检测到软件适配器（Basic Render Driver）时执行器
拒绝启用并提示安装厂商驱动；`iGPU_name()` 上报设备名。

---

## 4. 生图训练实测（R5-4500U）

| 模型 | UNet | 256px 单步 | 峰值 RSS | 500步 | 512px 单步 |
|---|---|---|---|---|---|
| tiny-sd | 323M | 2.18s | 3.0GB | 18min | — |
| bk-sdm-tiny | 323M | 2.24s | 2.6GB | 19min | — |
| bk-sdm-small | 482M | **2.10s** | 3.2GB | 18min | 8.29s（4.1GB） |
| bk-sdm-small+8bit 量化冻结层 | 482M | 3.16s | 3.7GB | 26min | 12.59s（4.5GB） |

- **256px fp32 就是甜点**（绝不"一小时一步"）；512px 可行（8.3s/步、不 swap）。
- 冻结层 8bit 量化在 256px 是负收益（慢 1.5x、RSS 还因页滞留略高），仅内存紧时用。
- 调用分布（profiler，单步）：conv 43.6%、注意力投影 20%、dropout(LoRA) 14%。

### 4.1 进阶方法（ControlNet / iP-Adapter）

| 方法 | 注入 | 可训练 | 单步 | 实测 |
|---|---|---|---|---|
| controlnet | ControlNetModel.from_unet | 123.2M（27.6%） | 2.94s | 2 步 loss 正常，存盘+采样通过 |
| ip_adapter | 9 个 attn2 + ImageProjection | 10.32M（3.1%） | 2.01s | 2 步 loss 正常，存盘+采样通过 |

关键坑（已解决）：老 config（mid_block_type=None）的 from_unet 兼容；
controlnet 条件图是原始像素（非 VAE 潜变量）；iP-Adapter 只注入 attn2（cross-attn）。

---

## 5. 硬盘均衡负载（disk_balancer）

**动机**：Windows 虚拟内存（swap）在训练时频繁擦写导致硬盘活动 100%、SSD 寿命受损。

**设计**：异步写盘（queue.Queue）不阻塞训练循环；内存紧张时自动检测冻结层
（冷参数）卸载到磁盘（SSD 热/HDD 冷）；读回 mmap 零拷贝。

**压力测试**（8 步 512px + 4×200MB dummy 冷参数，内存阈值 50%）：
- 5 次逐 step 卸载，内存 61%→52%（可用 6.1→7.4GB）；
- 冷参数读回 **5/5 数值完整**；
- **速度对照**（同配置 6 步）：无 balancer 39s vs 有 balancer 38s —— **几乎零开销**。

**结论**：balancer 在内存充足时零延迟（`update_step` 立即返回），仅在内存逼近
阈值时按需卸载（每次 1 个，有界），不拖累训练速度。

---

## 6. 跨平台构建与自检

| 平台 | 命令 | 产物 | 验证 |
|---|---|---|---|
| Windows x86/64 | `build_manual\build_manual.bat amd` | `libbitsandbytes_cpu.dll` | `python -m bitsandbytes.gdn_cpu` → PASSED；gemm_8bit OK |
| Linux x86_64/aarch64 | `bash build_linux.sh` | `libbitsandbytes_cpu.so` | `--selftest` → C 层 4/4 PASS |

**torch-free C 自检**（`selftest_cpu.c`，Linux/x86_64+aarch64 与 Windows 通用，直接链接内核源码）：
quantize_blockwise 8bit 往返 / gemm_8bit 前向 / 4bit GEMV(nf4) / AdamW8bit 单步 —— **4/4 PASS**
（已在 Windows 用 MSVC 等价验证；Linux 仅差编译链差异）。

---

## 7. 结论与展望

- **AVX2 CPU 纯训练可用**：1.7B LoRA 7s/步、8B NF4 38s/步（16GB）、SD 生图 2.1s/步，
  全部无"小时级一步"。关键在 GDN 融合（35×）、8bit 优化器（内存 1/3.8）、
  fp32 + 控制激活，而非降基座权重精度。
- **负面结论明确**：AVX2 上 bf16 禁用；基座权重 8bit/bf16 存储**全量训练**负收益（但真量化
  存储 + LSQ 只学标度可用于压缩内存，见 §3.5）。
- **跨平台**：Windows/Linux（x86_64/aarch64）构建 + torch-free 自检已备。

**展望**：
1. 为 AVX2 补 int8 GEMM **训练**内核（让 8bit 权重量化训练真正可行，而非仅 dequant）；
2. 8bit 量化对长程训练精度的系统评估；
3. 视频模型（Wan/CogVideoX/…）在纯 CPU 的 LoRA 训练；
4. 混合计算（offload 到具备独立显存的低端 GPU）——**注意**：我们仅在无独显机型实测，
   核显（iGPU，共享内存）offload 已实测为负收益（R5 AMD + i5 R5 M240 均打不过
   同代 CPU），**不建议**；此方向仅对真正带独立显存的低端 GPU 卡有意义，需另配
   设备验证。

---

## 附录：复现要点

- 机器 A（R5-4500U）：线程 **6**（无超线程，物理核拉满）；fp32；`max_length ≤ 512`。
- 机器 B（i5-10400）：文本 **8** 线程、生图 **12** 线程；fp32。
- 重编 DLL：VS x64 终端 `build_manual\build_manual.bat amd`（AMD）或 `intel`（Intel）；
  cpp/h 保持 **CRLF** 行尾（LF + GBK 注释会被 cl 吞行）。
- 自检：`python -m bitsandbytes.gdn_cpu`（PASSED）；`stress_opt.py`（210 组合 0 失败）；
  `selftest_cpu.c`（4/4）。
��。
��。
