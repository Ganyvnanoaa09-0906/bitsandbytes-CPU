# 技术文档：爆改版 bitsandbytes CPU 后端

> 面向对象：技术人员（读得懂 C++/PyTorch 内核的开发者）
> 目的：说明本 fork 在 bitsandbytes v0.45.1 基础上**改了什么、为什么改、怎么用**。

---

## 1. 背景与动机

上游 bitsandbytes 的 CPU 后端（`backends/cpu/`）主要面向**推理**：
`quantize_blockwise` / `dequantize_blockwise` / 4-bit GEMV 等算子以
「dequant 到 fp32 → oneDNN/MKL GEMM」为默认路径。在纯 CPU 机器
（无 NVIDIA GPU、仅 AVX2 或 ARM64 NEON）上做**训练**时，存在三个硬伤：

| 问题 | 后果 |
|---|---|
| 基座权重以 fp32 存储 | 1.7B 模型 6.8GB，叠加梯度+优化器状态后 12/16GB 机器必 swap |
| dequant-then-GEMM | 每步把全部权重 dequant 成 fp32 临时张量（1.7B≈6.8GB 临时），带宽翻倍 |
| 无训练级 8bit 优化器 | 优化器状态 4 字节/参数（Adam：m+v=8 字节），内存吃紧 |
| Gated DeltaNet（GDN）慢路径 | Qwen3-Next/3.5 的线性注意力在 CPU 上退化为逐时间步 Python 循环，一次反向 728 秒 |

**本 fork 的目标**：在这些 AVX2 / NEON 机器上，把「LoRA 训练、量化冻结层、
对齐（DPO/KTO/…）、SD 生图训练」从"不可用"提升到"可用"（秒级/步）。

---

## 2. 改动总览（相对 v0.45.1）

```
csrc/
  cpu_gdn.cpp          [新增] Gated DeltaNet 前向/反向融合内核（AVX2/FMA，OpenMP 按 head 并行）
  cpu_ops.cpp          [扩展] 8bit/4bit 量化、宏内核、8bit 优化器融合、gemm_8bit 反量化 GEMM
  cpu_ops.h            [扩展] 新增内核声明（gemm_8bit 等）
  pythonInterface.cpp  [扩展] c* CPU 符号导出（extern "C"）
bitsandbytes/
  gdn_cpu.py           [新增] GDN Python 封装 + patch_transformers/patch_fla + selftest
  _ops.py              [新增] bitsandbytes::gemm_8bit op 定义 + fake
  backends/cpu/ops.py  [修改] gemm_8bit CPU 注册 + 4bit GEMV 回退修复
  functional.py        [新增] fused_dequant_linear_8bit API
build_manual/          [新增] Windows MSVC 手动编译（绕过 CMake）+ export.def
build_linux.sh         [新增] Linux g++/clang 编译
selftest_cpu.c         [新增] 无 torch 的 C 层内核自检
tests/test_cpu_e2e.py  [新增] CPU 端到端验证
```

---

## 3. 各内核技术细节

### 3.1 GDN 融合内核（csrc/cpu_gdn.cpp）

**解决的问题**：Qwen3-Next/3.5 的 Gated DeltaNet 线性注意力，在无 Triton 的
CPU 上经 Transformers 慢路径转成逐时间步（per-step）Python 循环，
反向图深度 O(T)，T=1024 时单层反向 728s，训练不可用。

**实现**：
- AVX2/FMA 向量化：`beta_k`、`input_gate`、`output_gate`、状态更新全在寄存器内；
- OpenMP 按**独立 head**（batch×head）并行——head 之间无依赖；
- 反向用**分块检查点**（chunk checkpointing）：内存复杂度由 O(T) 降为 O(⌈T/C⌉)；
- 任意线程数：输出逐位一致（同步规约顺序固定）。

**实测**（i5-10400，T=1024, B=2, H=4, K=V=64）：
`fused 51.6ms vs 朴素逐时间步 1813ms` → **≈35×**；
反向精度（fp32）相对 double 参考 ~1e-7。

### 3.2 8-bit 块量化/反量化（cpu_ops.cpp）

- **量化**（`cquantize_blockwise_cpu_*`）：LUT 加速的最近码搜索。8bit 用
  **线性 code map**（`code[i]=2i/255-1`）。注意：默认 `create_dynamic_map()`
  是零点加密的非线性映射，**无法用标量 FMA 折叠解码**——gemm_8bit 需配合线性 map。
- **反量化**（`cdequantize_blockwise_cpu_*`）：AVX2 查表展开，4096² 实测
  10.6ms（优化前 17.6ms，快 40%）；4-bit NF4/FP4 走 pshufb/NEON LUT。

### 3.2.1 量化基座直接训练（quant_lora.QuantLinearTrainable，R7）

**目标**：把原「STE QAT 伪量化（fp32 master 驻留、不减内存、已知负收益）」的
`quant_base` 改为**真量化存储 + LSQ 可学习标度**，实现真正的权重内存压缩。

**与旧实现的区别**：基座权重**真正存成 8bit/NF4/FP4 码字**（`register_buffer("wq")`
+ `register_buffer("code")`），**fp32 master 权重不再驻留**：

| 精度 | 每元素字节 | 权重内存 | 说明 |
|---|---|---|---|
| 8bit | 1 B | 省约 75% | blockwise 量化 |
| NF4/FP4 | 0.5 B | 省约 87.5% | 打包 4bit |

**训练机制（LSQ，Learned Step-size Quantization）**：
- 仅把 `scale`（每 block 一个标量）作为可学习的 `nn.Parameter`（初值 = 量化时
  `state.absmax`）；
- `forward`：`w = code[wq] * scale` —— 纯 PyTorch 查表 + 广播，无需调用 C 内核；
- `backward`：梯度自然流回 `scale`，**无需对离散码字做 STE**（码字本身冻结）；
- 优化器只维护 `scale`（每 block 一个标量，量级极小）。

**边界**：
- 只调标度、不调码字（量化码字冻结）。若需**同时更新基座权重与低秩增量**，
  请改用 `qlora`（量化冻结基座 + fp32 LoRA）；
- `forward` 仍是「反量化到 fp32 再 GEMM」，**未走 gemm_8bit 融合**（纯 PyTorch 路径，
  主要收益是**权重内存压缩**）；若后续需加速前向，可把 `_dequant` 接驳到
  `gemm_8bit` 融合内核；
- 提供 `quant_weight_bytes()` 统计本层实际占用（码字 + 标度 + 码本）。

**入口**：`from quant_lora import QuantLinearTrainable`；上层统一入口为
`train.py --method quant_base`（经 `peft_backends._apply_quant_base` 替换 `nn.Linear`）。

**实测**（`jiaoben\test_quant_base.py`，PASSED）：
8bit 权重 48.0KB → 13.2KB（省 72.5%，重标度 max|Δ|=0，scale 梯度 120.77）；
NF4/FP4 48.0KB → 6.8KB（省 85.8%，scale 梯度 126.32 / 141.14）。

### 3.3 8-bit 优化器融合内核（coptimizer_update_8bit_blockwise_cpu）

把 `kOptimizerStatic8bit{1,2}StateBlockwise` 移植为 CPU 单遍融合内核：
**dequant → 更新 → p 更新 → requant** 一次内存扫过，覆盖
Adam/Lion/RMSProp/AdaGrad/Momentum/AdEMAMix（通过 `optimizer_id` 分发）。

- 8bit 状态内存 = fp32 的 **1/3.8**（m、v 各 1 字节/参数 + absmax）；
- 标量与 AVX2 路径逐位一致；`skip_zeros` 与 CUDA 语义对齐；
- 实测 4M 参数单步 13.1ms；与 fp32 收敛终态差 <0.01。

### 3.4 8-bit 融合反量化 GEMM（gemm_8bit，本 fork 特色）

```cpp
// out[M,N] = A[M,K] @ dequant8(B[N,K])^T
void cgemm_8bit_inference_cpu_fp32(A, B_uint8, absmax, out, M,N,K, lda,ldb,ldc, blocksize);
```

**为什么不是「dequant 后再 GEMM」**：
- dequant-then-GEMM 需要把整块权重先变成 fp32 临时张量（如 1.7B→6.8GB 临时），
  且一次额外读写全量权重；
- 融合内核让权重**全程保持 uint8**（DRAM 流量 = fp32 的 1/4），在寄存器内解出
  浮点值参与 FMA：`w = (code·(2/255) - 1)·s = code·(2s/255) - s`（一次 FMA 折叠）；
- 每输出列共享 B 行解码（m 以 4 行一块摊销），AVX2 8 宽 FMA。

**精度**：与 dequant+F.linear 相比相对误差 ~4e-7（仅 fp32 求和顺序差异）。
**适用**：冻结线性层权重 8bit 存储的推理前向 / 训练时 `dx = dout @ dequant(w)`。

**约束**：`K % blocksize == 0`（否则回退 scalar）；用**线性 code map** 量化。

### 3.5 4-bit 推理 GEMV 回退修复

上游 `backends/cpu/ops.py` 的回退路径此前调用仅在 `AVX512+BF16` 构建下编译的
符号 `gemv_4bit_inference_cpu_fp4/nf4_bf16`，AVX2 机器上直接 AttributeError；
且 nf4 分支传 `data_type=0`，命中内核 `data_type!=FP4 && !=NF4` 直接返回
（输出全 0 且无报错）。本 fork：
- 改为调用所有构建导出的 `cgemv_4bit_inference_cpu_{fp32,bf16,fp16}`；
- `data_type` 按内核常量：**FP4=1 / NF4=2**；
- 修复后 nf4/fp4 与参考实现逐位一致（误差 0）。

### 3.6 GDN 接入 Transformers（gdn_cpu.py）

- Transformers 5.15 中 Qwen3-Next/3.5 的 GDN 慢路径符号为
  `torch_recurrent_gated_delta_rule` 与 `torch_chunk_gated_delta_rule`
  （而非旧版 `fused_recurrent_gated_delta_rule`），模型 forward 动态查询；
  `patch_transformers()` 同时替换这三个符号；
- 语义对齐：慢路径在函数内部对 query 乘 `1/sqrt(K)`，而融合内核默认不缩放，
  在 dispatch 层补缩放；顺序改为「先 L2 归一化、后缩放」；
- 非 CPU 张量 / 非 GDN 场景保持原行为；varlen（`cu_seqlens`）不支持时回退原路径。

### 3.7 EFST：MoE 专家专项微调（efst.py）

**目标**：MoE 模型在纯 CPU / 低内存环境下微调——只训练被选中的专家（及可选 router），
冻结其余全部参数。冻结参数不产生梯度、不占优化器状态，直接压缩内存。

**核心机制**：
- `freeze_all` + `unfreeze_experts`：只解冻 `expert_indices` 指定的专家（可手动，或用
  校准数据 `collect_expert_usage` / `select_top_experts` 按路由热度自动选 `top_k` 个）；
- `add_lora_to_experts`：给选中专家注入 LoRA，把可训练量压到 adapter 级别；
- 自动识别常见专家结构（`experts` / `moe` / `block_sparse_moe`）。

**支持的专家结构与注意点（已实测）**：
- **经典 ModuleList/ModuleDict**：专家容器（如 `mlp.experts`）的**子模块即专家**，
  各专家独立子模块 → 可逐专家解冻 + 注入 LoRA。实测（8 专家模型，top_k=3）：
  `find_expert_groups` 识别出 8 专家、自动选最热 3 个、**冻结其余 5 个**、可训练
  34856→4872（14%）且 loss 收敛。
- **3D 张量专家**：见下方专节（Qwen3Next/Qwen3.5）。
- ⚠️ **嵌套 gate+experts 的自定义 block**（如 `TinyMoEBlock` 内部含 `gate` + `experts`
  ModuleList）：`find_expert_groups` 会把**整个 block** 当专家容器、把 `gate` + `experts`
  两个子模块当成 2 个"专家"，**不会展开 `experts` ModuleList 到 8 个**。若模型是
  标准 MoE（`mlp.experts` = 专家 ModuleList）则无此问题；已实测标准结构 EFST 完全正确。

**3D 张量专家支持（transformers 5.15+ Qwen3Next/Qwen3.5）**：
新版 MoE 专家权重是 `[num_experts, ...]` 大 Parameter（`Qwen3NextExperts` 的
`gate_up_proj` / `down_proj`），无独立子模块。EFST：
- **识别**：类型名含 expert/moe 且持有 3D 参数 → 建 tensor 专家组；
- **路由统计**：用 forward 的 `top_k_index` 逐 token 计数，按专家粒度选热专家；
- **按行解冻**：`split_3d_expert_params` 将 3D 张量拆成 `ParameterList`（PyTorch
  requires_grad 是参数级，不拆无法按行冻），并把 forward 换成逐专家循环版；
- **LoRA 回退**：tensor 专家无 `nn.Linear` 子模块，无法注入 LoRA —— `lora=True`
  时对 tensor 专家**自动回退为直接解冻选中专家行**（经典专家仍走 LoRA）；
- **注意**：拆分后 `state_dict` 键从 `gate_up_proj` 变为 `gate_up_proj.0`… ——
  先 `apply_efst` 再加载权重，保持键一致。

**实测**（i5-10400，随机初始化 Qwen3Next 3 层混合模型）：
3 个 tensor 专家组 × 8 专家、top-2 选择 → 可训练参数 725,712 → 143,600（19.8%），
60 步训练 loss 下降 23% 且收敛正常。

**入口**：`from efst import EFSTConfig, apply_efst`；配合
`bitsandbytes.gdn_cpu.patch_transformers()` + `bnb.optim.AdamW8bit(...)` 使用
（8bit 优化器只给可训练参数分配状态）。

### 3.8 核显（DirectML）块级常驻执行器（gpu_scheduler.py，实验性）

**设计动机**：无独立显存机型（R5-4500U 共享 DDR4-2667，实测可用带宽约 21~22 GB/s，
约为理论值 42.7 的一半）上，带宽调度没有增量可挖；唯一有效的形态是**减流量 + 少
同步**：冻结基座权重常驻核显（每步 0 权重传输）、前向+反向整步在核显上执行、每步
只同步一次。

**关键实验结论（决定架构）**：

| 形态 | 结果 |
|---|---|
| 逐算子调度（每算子排队+排空） | 慢约 2 倍 —— `torch_directml` 每次 `.to("cpu")/item()` 是**整队列排空**，单次约 10~15ms |
| 大 GEMM 流水线（50×2048²，末尾同步一次） | 294.5 GFLOPS vs CPU 226.7 = **1.30×** |
| 真实 Qwen 结构全链 seq=128（权重常驻） | **1.35×** |
| 同全链 seq=512 | 0.70×（DML 的 `F.sdpa` 长序列 3~7× 慢于 CPU；`multi_head_attention` 专用内核需新驱动，本机 27.20.11032 不支持） |

**实现要点**（`IgpuExecutor`，`train.py --igpu` 的底层）：
- `prepare(tokens=)`：前置检查（DirectML 可用 / 全 fp32 / 基座内存 ≤ 上限 / 每步
  token ≤ `GPU_SCHED_MAX_TOKENS` 默认 256 / 启动时 GEMM 校准达标），成功后
  `model.to(DML)` 整体常驻，并为可训练参数建立 CPU 镜像；失败返回原因、调用方回退
  纯 CPU；
- `grad_to_cpu()`：可训练参数梯度拷回 CPU 镜像（供 bnb 8bit 优化器），并清空核显侧
  梯度——这是训练中唯一的 DML 同步点；
- `weights_from_cpu()`：优化器更新后的权重拷回核显；两处往返量仅适配器规模，可忽略；
- 调用方（train.py）每步末尾执行一次 `loss.item()`，即整队列排空点；每步只允许一次。

**换机自适应（Intel UHD 630 等，R8）**：不同核显算力差异极大，固定参数无法跨机复用，
故执行器默认自适应，无需逐台手调：
- **内存上限按整机内存等比缩放**：16GB→7000MB、12GB→5250MB、8GB→3500MB
  （`min(7000, RAM×7/16)`；核显共享内存 + DML 运行时约 0.5~1GB 额外开销，
  小内存机型必须收紧；显式 `--igpu_mem_mb` / `GPU_SCHED_MEM_MB` 可覆盖）；
- **启动时 GEMM 校准**（`calibrate_gpu()`）：用 2048² 流水线 GEMM（两轮取较好、
  长预热让核显爬频——单轮实测有 ±0.1 抖动）实测核显/CPU 吞吐比，低于
  `GPU_SCHED_CALIB_MIN`（默认 1.10，`--igpu_min_gain 0` 跳过）自动回退并打印实测值。
  本机（Vega 6）稳态校准 1.2~1.5x → 启用；UHD 630 预期低于门槛 → 自动回退，
  即「弱核显机器开 --igpu 无害，只是自动不生效」。换机自查：
  `python gpu_scheduler.py` 末行直接打印校准值与判定。

**CPU/核显异步并发：技术成立、训练负收益（R9，收关）**：torch-directml 为
「排空时才执行」的懒执行模型（入队 ~0.3ms，CPU 空转 800ms 后排空仍需完整执行
时间），普通「入队后 CPU 干活」天然串行；唯一能触发真并发的形态是**后台排空
线程**（主线程入队 + 后台线程 `.cpu()` 排空 + 主线程并发算 CPU，混合负载实测
1.32×，两端共享 DDR4 各劣化 ~1.6×）。但用于训练（微批数据并行半 CPU 半 DML）
实测 0.83× vs 常驻——本机 DML 快于 CPU，分给 CPU 的批量连带宽争用一起坐上
关键路径，最优分配＝全部给 DML（即常驻执行器）。非主线程入队为竞态行为
（曾触发裸 RuntimeError），不可依赖。探针：`jiaoben/igpu_async_probe.py` /
`igpu_async_drain_probe.py`。

**普适准则**：CPU/核显并发只在**两设备算力接近**、或**负载天然分属各自强项**时才有正
收益；若一设备明显更快（本机 DML ≈ 1.6× CPU），最优分配＝**全部给快设备**（常驻执行器），
任何「半 CPU 半核显」都会让慢的那半连同带宽争用坐上关键路径。适用于任何「主快副慢」
的异构加速：先测两设备吞吐比，明显 >1（如 1.2~1.5×）就别拆给慢设备；接近 1 或某类算子
本就是慢设备强项时才值得异步，且必须块级/整步一次同步。

**生图（SD）训练不用核显（R8 实测，方向关闭）**：SD UNet 主力算子在 DML 上全面
落后——conv3×3 主力形状 0.50~0.90×、S=1024 自注意力 0.14×（7 倍慢）、投影 GEMM
0.38~0.65×（oneDNN 的 conv/小 GEMM 在 6 核 AVX2 上太强，弱核显无算力优势）；
且 torch-directml 与 diffusers 0.40 存在版本墙（见下条），全链 DML 反向还会触发
插件崩溃。生图训练维持纯 CPU（fp32）路线。

**torch-directml 版本墙（torch241_compat.py）**：安装 torch-directml 会把 torch
降级到 2.4.1（其硬性依赖），导致 diffusers 0.40 导入即崩（字符串注解的
`infer_schema`、缺 `flex_attention`、`sdpa` 无 `enable_gqa` 三处不兼容）。项目提供
`torch241_compat.py` 运行时兼容层（幂等，torch≥2.5 自动跳过），凡装了
torch-directml 的环境跑 SD/生图脚本必须在 `import diffusers` 之前引入。
另注意：DML 上「先 `model.to(DML)` 再改 `requires_grad`」会**静默丢梯度图**，
必须先设 requires_grad 再迁移。

**算子兼容性**（DML autograd 实测矩阵，独立进程防队列死锁级联）：
matmul / gelu / `F.softmax` / `F.rms_norm` / SDPA / 因果 `masked_fill` / dropout /
cat / 切片 / RoPE / embedding 查表+反向 —— 全部可用；
- ⚠ `F.layer_norm` 反向仅支持「叶子输入」（贴在 matmul 后必挂）→ 需手搓
  `(x-μ)/√(σ²+ε)·γ+β`（GPT2 系模型注意；Qwen 系用 RMSNorm 无此问题）；
- ⚠ embedding 反向走 `index_add` CPU fallback（可用，有性能提示）；
- ⚠ 手搓 softmax 中的 max 反向撞 DML scatter 限制 → 用 `F.softmax` 即可。

**界限**：量化基座（QuantLinear 等）与 `--flash`（disk_balancer）组合暂不支持；
小模型（H<1024）与长序列（seq>256）为负收益区间，执行器自动拒绝并回退纯 CPU；
生图（SD）训练为已证实的负收益 + 环境不兼容方向，不走核显（见上）。

**设备兼容性**：本执行器基于通用 DirectML（D3D12）接口，已在 AMD Radeon(TM)
Graphics（R5-4500U）实测；Intel UHD 630（i5-10400 等）走同一接口，但核显相对
CPU 更弱，预期收益低于 AMD 机型。`prepare()` 会在检测到软件适配器
（Microsoft Basic Render Driver）时拒绝启用并提示安装厂商驱动；经 `iGPU_name()`
上报设备名（`description()` 输出）。换机自查入口：`py -3.11 gpu_scheduler.py`
（打印设备名与可用性）。

---

## 4. 内存策略：为什么「量化冻结层」而不是「降权重存储精度」

| 方案 | 权重内存 | 5 步耗时（1.7B+i5） | 结论 |
|---|---|---|---|
| fp32（基准） | 6.8GB | 36.8s | 基准 |
| 8bit 量化 + 前向 dequant | ~1.8GB | 375s（≈10×） | 负收益：逐层转换开销 + 量化期峰值内存 → swap |
| bf16 存储 + .float() | ~3.4GB | 1058s（≈29×） | 负收益：每步全权重转换 |
| **fp32 + 控制激活 + 8bit 优化器** | — | 基准 | **推荐** |

结论：AVX2 CPU 上**拒绝**降基座权重精度做训练；内存优化靠
「控制 max_length/batch + 8bit 优化器 + （可选）冻结层量化」。
gemm_8bit 为「真 8bit 冻结层、零 fp32 临时」提供了可能（详见第 3.4 节）。

---

## 5. 构建与测试

| 平台 | 命令 | 产物 |
|---|---|---|
| Windows x86/64 | `build_manual\build_manual.bat amd`（VS x64 终端） | `bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll` |
| Linux x86_64/aarch64 | `bash build_linux.sh` | `bitsandbytes\libbitsandbytes_cpu.so` |

**自检**（无 torch，内核级）：
```
bash build_linux.sh --selftest
# 或直接编译运行: clang/g++ -I csrc selftest_cpu.c csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp
# 覆盖: quantize 往返 / gemm_8bit / 4bit GEMV / 8bit 优化器
```

**完整自检**（需 torch）：
```
python -m bitsandbytes.gdn_cpu    # 应输出 selftest PASSED
python stress_opt.py             # 210 组合 0 失败（LLM 侧回归）
```

---

## 6. 已知约束与坑

1. **AVX2 机器禁用 bf16**：bf16 GEMM 由 oneDNN 软件模拟，单次 >90s 近乎卡死——训练务必 fp32。
2. **gemm_8bit 与线性 code map**：默认 `create_dynamic_map()` 是非线性映射，无法 FMA 折叠；8bit 量化请传 `code=torch.arange(256)*(2/255)-1`。
3. **换行符（LF vs CRLF）跨编译**：`.gitattributes` 强制 C/C++/sh 为 **LF**（clone 后自动 LF）。MSVC `cl` 在 GBK 代码页下解析 LF 源的中文注释会吞行——**Windows 编译必须在 `build_manual.bat` 加 `/utf-8`**（已内置）；Linux `g++` 天然处理 LF。LF + `/utf-8` = 跨平台一致且稳定。
4. **8bit 优化器 state 需初始化**：首次使用前 state 码与 absmax 需有效（0 状态 = absmax 全 0），否则产生 NaN。
5. **4bit GEMV 的 ldb**：packed B 的行距为 `K/2`，不是 K。
6. **非 AVX2 CPU（实验性）**：所有 AVX2 内核都有运行时 `has_avx2_cpu()`（CPUID）保护 + `BNB_CPU_NO_AVX2=1` 环境变量强制关。非 AVX2 机器运行时走 scalar fallback——**能跑（慢），但不保证性能**。`build_linux.sh` 会自动检测（无 AVX2 则编译 `-march=x86-64` 不定义 `__AVX2__`）。**实验性：无 AVX2 设备实测，理论可跑，遇崩设 `BNB_CPU_NO_AVX2=1` 或 `build_linux.sh --no-avx2`。**
7. **disk_balancer 在 WSL 里对宿主 NTFS 做 9P 卸载 = 严重风险（会扰 MFT，务必排除）**：在 **WSL** 上跑 disk_balancer（`--flash`）时，它的 **Linux 分支（`detect_disks`）会把宿主 Windows 的 NTFS 挂载（`/mnt/c`、`/mnt/d`，经**微软 9P 协议**）当成普通磁盘**，去对它们做冷参数卸载（SSD/HDD 读写）。**9P（网络文件系统语义）在高负载/逼近内存极限下对 NTFS 做高频读写/删除 → NTFS MFT 元数据经 9P 未正确落盘 → 文件系统异常 / 数据丢失**（实测案例：i5 WSL 生图高负载跑 `--flash -a` 后，宿主 D 盘疑似 MFT 紊乱、文件丢失、C 盘受影响）。**根因已定位：disk_balancer 的 Linux 分支未排除 WSL 的 9P/NTFS 挂载点。** **修复：`detect_disks`/`_get_disk_io` 必须排除 `9p`/`fuse`/宿主 NTFS 挂载**（fstype 非 `vfat`/`ntfs`/`9p`/`fuse`，mountpoint 非 `/mnt/[a-z]`），**绝不对宿主 NTFS（`/mnt/c`、`/mnt/d`）做冷参数卸载**；WSL 里遇到 9P/fuse 挂载应跳过/降级（不卸载冷参数，宁可不做也要安全）。**若已出现文件丢失**：① 立即停止 D 盘写入；② **先只读 `chkdsk <盘>:`（不带 /f）**；③ **不要 `chkdsk /f` / 格式化**（会把可恢复数据标记丢失）；④ 优先备份能读出的数据，数据恢复工具（TestDisk/Recuva）装 C 盘/另一盘运行、别写 D 盘。
8. **Qwen3/Qwen3.5 训练需 `transformers>=5.15.0`（否则 CPU backward 段错误）**：实测 **`transformers 5.6.0` + torch 2.13+cpu 在 R5 (AVX2) 上对 `Qwen3ForCausalLM` 做 `loss.backward()` 会触发 `Windows fatal exception: access violation`**（前向正常，纯 torch(不经过 bnb) 也一样崩——是 transformers 5.6.0 的 Qwen3 CPU backward bug，**与 bnb DLL 无关**）。**换 `transformers==5.15.0` 后 backward 恢复正常**。→ **环境要求：`pip install "transformers>=5.15.0"`**。验证脚本：`python tools/verify_train_release.py --model <本地模型目录>`（免联网、免 tokenizer）。注意：`llamafactory 0.9.5` 与 transformers>5.6.0 存在依赖冲突，若同时用 llamafactory 需单独评估版本。
