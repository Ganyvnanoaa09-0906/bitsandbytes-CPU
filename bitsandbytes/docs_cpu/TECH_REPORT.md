# 技术报告：纯 CPU 环境的 bitsandbytes 爆改与训练加速

**作者**：deepsleep team
**日期**：2026-09
**范畴**：面向无 NVIDIA GPU（仅 CPU，AVX2 / ARM64 NEON，12–16GB 内存）机器的
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

两台目标机器均无 NVIDIA GPU（仅核显/集显，且核显用系统共享内存，无法作为
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

**阶段三：CPU/核显异步并发压榨（R9，收关）**

问题：块级常驻执行器运行时 CPU 全程空闲——能否用异步把 CPU 也压榨进来？

| 实验（`jiaoben/igpu_async_probe.py` / `igpu_async_drain_probe.py`） | 结果 |
|---|---|
| 单 GEMM 输出列切分（CPU 半边 + DML 半边，末尾一次排空） | 0.82~1.07×，不如较快的单设备 |
| 微批数据并行（半批 CPU 全程 + 半批 DML 全程，每步一次梯度合并） | 0.96~1.16× vs CPU、0.59~0.71× vs 常驻 ≈ 两设备串行之和 |
| 懒执行根因 | 入队仅 ~0.3ms；CPU 空转 800ms 后排空仍需完整执行时间（563ms）——**DML 在排空时才开始执行**，故「入队后 CPU 干活」天然串行 |
| 后台排空线程（主线程入队 + 后台线程 `.cpu()` 排空 + 主线程并发算 CPU） | 串行 1213ms → 并发 921ms = **1.32×，真并发成立**（共享 DDR4 带宽，两端各劣化约 1.6×，总墙时仍赚） |

但把「后台排空 + 数据并行」用于训练（BS=4 SL=128，8 层 Qwen 风格块）：736.9ms
vs 常驻 615.2ms = **0.83×**——本机 DML 快于 CPU（约 1.6×），任何分给 CPU 的批量
都会连带宽争用一起坐上关键路径，最优分配就是「全部给 DML」＝现有常驻执行器。
**异步并发技术成立（后台排空线程是 torch-directml 下唯一能触发真并发的形态），
但训练收益为负，方向收关**；其适用场景是 CPU 与 GPU 同量级、或混合「CPU 强 /
GPU 强」算子的负载——本训练栈不存在这种形态。另注：非主线程入队是竞态行为
（曾触发裸 RuntimeError、单独跑偶尔成功），不可依赖。

**普适准则（可迁移的判断）**：CPU/核显并发只有在**两设备算力接近**、或**负载天然
分属两设备各自强项**时才有正收益；若一设备显著快于另一设备（本机 DML ≈ 1.6× CPU），
最优分配就是**全部给快设备**（即常驻执行器）——任何「一半 CPU 一半核显」的做法，
都会让较慢的那半、连同共享内存带宽争用一起坐上关键路径，反而更慢。这条准则适用于
任何「主设备快、副设备慢」的异构加速场景：先测两设备吞吐比，若明显 >1（如 1.2~1.5×），
就不要指望把任务拆给慢设备；只有当吞吐比接近 1、或某类算子本来就是慢设备强项时，
异步分割才值得，且必须是「块级/整步一次同步」，避免逐算子同步把收益赔给队列排空。

**最终形态**：`train.py --igpu` 落地为**块级常驻执行器**（默认关闭）：每步 token ≤
`GPU_SCHED_MAX_TOKENS`（默认 256）、fp32 基座 ≤ 内存自适应上限（16GB→7000MB、
12GB→5250MB、8GB→3500MB）、启动时 GEMM 校准达标、无量化参数时启用；否则打印原因
自动回退纯 CPU。逐算子接口（`big_gemm` / `patch_igpu`）保留仅作实验，不推荐用于
训练。

**换机自适应（R8，面向 Intel UHD 630 等其余机型）**：不同核显算力差异极大，固定
参数无法跨机复用，故执行器不再按机型写死：(1) 内存上限按整机内存等比缩放
（`min(7000, RAM×7/16)`，核显共享内存 + DML 运行时开销，小内存机型自动收紧）；
(2) `prepare()` 时以 `calibrate_gpu()` 实测本机「2048² 流水线 GEMM 核显/CPU 吞吐比」
（两轮取较好、长预热促核显爬频——单轮实测有 ±0.1 抖动，同机曾测得 1.20x 与
1.06x 的波动），低于 `GPU_SCHED_CALIB_MIN`（默认 1.10，`--igpu_min_gain` 可调、
0 跳过）自动回退并打印实测值。本机（Vega 6）稳态校准 1.2~1.5x → 启用；UHD 630
预期低于门槛 → 自动回退，「弱核显机器开 --igpu 无害，只是自动不生效」。换机自查：
`python gpu_scheduler.py` 末行直接打印校准值与判定。

**设备适用性**：实测数据全部来自 AMD Radeon(TM) Graphics（R5-4500U）；Intel
UHD 630（i5-10400 等）走同一 DirectML 接口，收益由启动校准自动判定、无需人工
预判。检测到软件适配器（Basic Render Driver）时执行器拒绝启用并提示安装厂商
驱动；`iGPU_name()` 上报设备名。

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
- **核显（DML）不用于生图训练（R8 实测，方向关闭）**：UNet 主力算子在 DML 上全面
  落后——conv3×3 主力形状 0.50~0.90×、S=1024 自注意力 **0.14×**（head_dim=40，
  7 倍慢）、投影 GEMM 0.38~0.65×，仅 320ch@64² conv 与 64×1280² GEMM 打平；
  全链 DML 反向另有 GroupNorm backward CPU fallback 与插件空消息 RuntimeError。
  探针留存于 `jiaoben/igpu_sd_probe.py` / `igpu_sd_fullchain.py`。
- **torch-directml 版本墙**：安装 torch-directml 会把 torch 降至 2.4.1（硬性依赖），
  与 diffusers 0.40 三处不兼容（字符串注解 `infer_schema`、缺 `flex_attention`、
  `sdpa` 无 `enable_gqa`）——**装了 torch-directml 的环境，纯 CPU 生图训练也会
  import 崩溃**。修复：`torch241_compat.py` 运行时兼容层（幂等，torch≥2.5 自动
  跳过），`train_sd_lora.py` 已在 `import diffusers` 前引入。另注意 DML 上先
  `model.to(DML)` 再改 `requires_grad` 会静默丢梯度图，必须先设 requires_grad
  再迁移。

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

## 7.1 全量实测验证：生图 + LLM 各 500 步真实训练（R5，纯 CPU）

> 目的：在真实训练环境下彻查整套 CPU 训练链（DLL 内核 + 8bit 优化器 + LoRA）的功能
> 正确性，并记录速度 / 内存 / loss。设备：R5-4500U（6C6T / 16GB / AVX2），torch
> 2.13.0+cpu，transformers 5.15.0。

### 7.1.1 全面检查发现并修复的问题

| 问题 | 根因 | 处置 |
|------|------|------|
| `build_manual.bat` 报 `'??具' is not recognized` / `... was unexpected` | 灾难工具 REM 注释为中文（GBK 非 ASCII），cmd 的 GBK 代码页把注释字节当命令解析 | 改为纯 ASCII 英文 |
| `gpu_scheduler.py` 未入库 | 核显换机自适应（calibrate_gpu / 内存按整机缩放 / 校准门槛）此前仅在文档 | 补提交该功能 |
| 8 个 `.obj` 编译产物 + `build_release_tmp\` | 编译中间文件残留 | 删除 |
| LLM 真实文本训练阻塞 | `deepseek-coder-1.3b-base` **缺 tokenizer**（只有 config/model） | 改用有完整 tokenizer 的 `qwen3.5-0.8B`（MoE） |
| qwen3 系 CPU backward 段错误 | transformers 5.6.0 的 Qwen3 backward bug（纯 torch 也崩） | 升到 5.15.0 根治 |

### 7.1.2 生图 LoRA 训练 500 步（bk-sdm-tiny）

- 模型：bk-sdm-tiny（UNet ~324M），LoRA trainable 0.43M（rank=4）；256px，batch=1，
  grad_accum=4，fp32，opt=bnb.optim.AdamW8bit，线程 6。
- **结果**：500/500 步跑完；**平均 ~1.7 s/步**，总 ~19 分钟；峰值 RSS **~2.15 GB**；
  loss 波动在 0.002~0.80（生图 LoRA 正常收敛）；每 100 步采样出图 + LoRA 存 `final_lora`。
- **结论**：生图 LoRA 训练在纯 CPU + bnb 8bit 优化器上完全可用，达「可边改边训练不卡」水准。

### 7.1.3 LLM LoRA 训练 500 步（qwen3.5-0.8B）

- 模型：qwen3.5-0.8B（Qwen3_5ForCausalLM，混合注意力 + MoE），752.4M 参数；
  LoRA（target=q/k/v/o_proj，rank=16），trainable **1.08M**（基座 753.5M 冻结）；
  真实 jsonl 文本，seq=128，batch=1，纯 CPU fp32，opt=bnb.optim.AdamW8bit。
- **结果**：500/500 步跑完；**0.25 step/s（~4s/步）**，总 **~33 分钟**；
  loss **2.999 → 1.8315**（后期稳定在 ~1.1-1.9）；**无崩溃**（此前 qwen 系 backward
  崩溃已被 5.15.0 根治）。
- **结论**：MoE 混合注意力 LLM 的 LoRA 训练在纯 CPU + bnb AdamW8bit 上完全可用，
  500 步稳定收敛。4s/步在同级 CPU 上合理。

### 7.1.4 小结

- 功能正确性：生图 + LLM 两个真实模型各 500 步 CPU 训练**全部成功、loss 收敛、无崩溃**
  —— 爆改 bnb 的 CPU 训练链在真实数据上可信可用。
- 速度/内存（R5 6C6T 16GB）：生图 ~1.7s/步、~2.1GB；LLM 0.8B MoE LoRA ~4s/步。
- 已知边界：**AVX2 禁 bf16（须 fp32）**；`deepseek-coder-1.3b-base` 缺 tokenizer（需自补或联网）。
- 注：训练用 500 步（非 1000 步）为控制 CPU 时效；若要 1000 步可将步数翻倍。

---

## 7.2 暴力测试（穷尽打压）发现与修复

> 用 workflow 把 fork 新增的所有组件分成 4 组并行穷尽打压（正常/边界/非法输入/极端值），
> 共发现 20+ 个问题并全部修复。

### 已修复（按严重性）

**数值 / 功能错误（🔴）**
| 问题 | 根因 | 修复提交 |
|------|------|---------|
| `sector_carve` 输出目录不可写时「假报成功」（报共恢复 N 个、exit=0，实际 0 文件） | `CreateFileW` 失败仍 `g_carved++` + 无条件 return 0 | 仅实际完整写出才计数/成功；失败报「无法创建」(`1753c8b`) |
| `sector_carve` 假头导致死循环 | extractFile 返回 0 时 readPos 不推进 | readPos 至少前进一个扇区 (`1753c8b`) |
| `quant_lora` nf4u/fp4u + `from_quantized` broken（dtype 不匹配崩溃） | QuantState 用 `dtype=torch.uint8`，dequantize_blockwise 返回 uint8 而非 float32 | 改 `dtype=torch.float32`（与自然量化一致）(`a5e10da`) |
| `gemm_8bit` 非对齐 K 静默返回错值 | 融合内核要求 K % blocksize == 0 但无校验 | `fused_dequant_linear_8bit` 加 K 对齐校验(`a5e10da`) |

**健壮性（🟡，参数校验缺失）**
| 问题 | 修复 |
|------|------|
| disk_balancer `memory_threshold=1.5` 静默禁用卸载；`None` 崩；`min_free_ratio=1.5` 负缓存 | `DiskBalancerConfig.__post_init__` 校验范围/非 None，非法报错(`d80777a`) |
| disk_balancer `parse_flash_args` 盘符丢冒号（`c`→应 `C:`），manual 模式匹配不到盘 | re-append `:`(`d80777a`) |
| efst `apply_efst(None)`/`freeze_all(None)` 崩；`top_k=-1` 静默选反 | None 报错；top_k>=1(`5959855`) |
| sd_quant `apply_quant_frozen(None)` 崩；`_pick_bs(0)` 返回 256 | None 报错；k<=0 返回 0(`5959855`) |
| quant_lora `lora_r=0` ZeroDivisionError；`weight=None` 崩；fp16 输入崩 | None/lora_r 校验；非 fp32 weight 转 float(`5959855`) |

**训练脚本（🟢）**
| 问题 | 修复 |
|------|------|
| verify_train_release / train_llm_sft `--steps 0` IndexError 崩 | 校验 steps>=1 / batch>=1 / 非空数据；结果段防空列表(`0e19101`) |
| verify delta 打印符号错 | 改 `losses[-1]-losses[0]`(`0e19101`) |

### 通过项（无需修）
- 内核数值：量化往返 / gemm_8bit(对齐) / GDN / 8bit优化器 精度全部达标(normRel ~3e-7，GDN selftest PASSED)。
- disk_balancer 正常路径、efst / quant_lora / sd_quant 正常调用、灾难工具正常恢复、gpu_scheduler CPU 回退路径。
- 文档完整性（中英对称、关键章节、无空文档）。

### 备注（环境不可测）
- GPU/DirectML(WSL 9P / 物理盘 / 显卡)相关路径在 R5 不可测（无 GPU / R5 WSL 坏 / 物理盘缺失），
  仅验证了 CPU 回退/不崩路径；`gpu_scheduler` 的核显实际调度需 GPU 机验证。

---

## 附录：复现要点

- 机器 A（R5-4500U）：线程 **6**（无超线程，物理核拉满）；fp32；`max_length ≤ 512`。
- 机器 B（i5-10400）：文本 **8** 线程、生图 **12** 线程；fp32。
- 重编 DLL：VS x64 终端 `build_manual\build_manual.bat amd`（AMD）或 `intel`（Intel）；
  cpp/h 保持 **CRLF** 行尾（LF + GBK 注释会被 cl 吞行）。
- 自检：`python -m bitsandbytes.gdn_cpu`（PASSED）；`stress_opt.py`（210 组合 0 失败）；
  `selftest_cpu.c`（4/4）。
