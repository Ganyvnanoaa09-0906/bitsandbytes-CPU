# 快速入门流程指南

> 适用对象：首次使用本项目的开发者。
> 说明：本文档提供每一步操作的**命令、命令作用及参数说明**，您可直接复制执行。

---

## 1. 前置条件

本项目是 bitsandbytes 的 **CPU 后端扩展**，面向**无独立显卡**（仅 CPU 计算、AVX2 或 ARM64 NEON 指令集）的机器，用于加速大模型的微调训练并降低内存占用。

### 依赖组件

| 组件 | 用途 | 安装方式 |
|---|---|---|
| Python 3.11 | 运行环境 | 官方安装包 |
| PyTorch（CPU 版本） | 张量计算核心 | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| 本项目 | 训练加速与内存优化 | 见「2. 编译」 |

> 注意：本环境无 NVIDIA 显卡，请安装 PyTorch 的 **CPU 版本**。若您使用 GPU 环境，本项目并非为 GPU 场景设计，不推荐使用。

---

## 2. 编译（Windows 平台）

在 Windows 上需**预先安装 Visual Studio**（勾选「C++ 生成工具」工作负载），随后打开 **x64 Native Tools 命令提示符**，执行：

```bat
cd /d <克隆路径>\bitsandbytes\build_manual
build_manual.bat amd
```

| 参数 | 取值 | 说明 |
|---|---|---|
| `amd` | 可选 | 强制使用 `/favor:AMD64`（针对 AMD CPU 的指令调度优化） |
| `intel` | 可选 | 强制使用 `/favor:INTEL64`（针对 Intel CPU） |
| （缺省） | — | 自动检测 CPU 厂商 |

**编译产物**：`bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll`。成功时输出 `[OK]`。

**编译验证**：

```bat
py -3.11 -m bitsandbytes.gdn_cpu
```

输出 `selftest PASSED` 即表示编译成功。

---

## 3. 环境配置

每次执行前（或置于脚本开头）：

```bat
set PYTHONPATH=<克隆路径>\bitsandbytes
set PYTHONIOENCODING=utf-8
```

---

## 4. 常用功能

### 4.1 量化冻结层（降低内存占用）

```python
from sd_quant import apply_quant_frozen
n, saved = apply_quant_frozen(model, quant_dtype='8bit',
                              exclude_names=('to_q','to_v','to_k','to_out'))
print(f"已量化 {n} 层，节省 {saved//1048576} MB")
```

**说明**：将不参与训练的层（冻结层）权重由 fp32 压缩至 8-bit，可降低约 75% 的显存/内存占用。

> 注意：`exclude_names` 必须包含**需要训练**的层（如 `fc2`）。若全部层均被量化，优化器将因参数列表为空而报错。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `quant_dtype` | 量化精度：`8bit` / `nf4` / `fp4` | `8bit` |
| `exclude_names` | 不参与量化、保持 fp32 的层名 | 空 |
| `cache` | 是否缓存反量化结果（`True` 提高速度但内存翻倍） | `False` |

### 4.2 8-bit 优化器（降低优化器内存）

```python
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-4)
```

**说明**：以 8-bit 存储优化器状态，占用约为标准 Adam 的 1/3.8。

### 4.3 8-bit 融合 GEMM（冻结层推理/训练）

```python
from bitsandbytes.functional import fused_dequant_linear_8bit, quantize_blockwise
import torch

code = torch.arange(256, dtype=torch.float32) * (2/255) - 1   # 线性 code map
q, st = quantize_blockwise(weight.reshape(-1), code=code, blocksize=256)
out = fused_dequant_linear_8bit(x, q.view(N,K), st.absmax.view(N,-1), 256)
```

**说明**：对 8-bit 量化权重直接执行矩阵乘法，全程不产生 fp32 大临时张量。计算式为 `out = x @ dequant8(w)^T`。

### 4.4 Qwen3-Next / Qwen3.5 快速内核（可选）

```python
from bitsandbytes.gdn_cpu import patch_transformers
patch_transformers()        # 必须在加载模型之前调用
```

**说明**：将 Qwen3-Next/3.5 中较慢的 Gated DeltaNet 计算替换为本项目提供的融合内核（约 35 倍加速）。

### 4.5 后台内存监视（保护 SSD）

```python
from torch_cpu_kit import start_mem_monitor, suspend_mem_monitor
ev = start_mem_monitor(interval=10)     # 每 10 秒检查一次
... 训练循环 ...
suspend_mem_monitor(ev)
```

**说明**：持续监控内存占用，当接近满载（可能触发磁盘 swap、频繁读写）时输出 `SWAP!` 告警。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `interval` | 检查间隔（秒） | 20 |

### 4.6 量化基座直接训练（真量化存储 + LSQ）

```python
from quant_lora import QuantLinearTrainable
import torch.nn as nn

lin = nn.Linear(128, 96)
q = QuantLinearTrainable(lin.weight, lin.bias, quant_dtype='nf4')
out = q(torch.randn(4, 128))   # 前向：查表反量化 + GEMM
out.sum().backward()           # 反向：梯度流回可学习标度 scale（LSQ）
```

**说明**：将基座权重真正存成 8bit/NF4/FP4 码字，fp32 master 权重不再驻留，仅学习每个
量化块的标度（LSQ，Learned Step-size Quantization）。收益是压缩权重内存——8bit 省约
75%、4bit（nf4/fp4）省约 87.5%；基座码字本身不更新。适用于内存紧张、且接受「只调标度
不调码字」的场景。本能力复用本仓库的 `quantize_blockwise` / `quantize_4bit` 与码本接口。

统一训练入口（上层项目）：

```bat
cd 那很有乐子了~
python train.py --method quant_base --base_model <path> --quant_dtype nf4
```

### 4.7 EFST：MoE 专家专项微调（省内存）

**用途**：MoE 模型在内存紧张时微调——只训练被选中的专家，冻结其余参数，
显著降低可训练量与内存占用。

```python
from efst import EFSTConfig, apply_efst

# 方式 A：用校准数据自动选最热的 top_k 个专家
config = EFSTConfig(
    top_k=2,                       # 自动选最热的 2 个专家
    calibration_dataloader=train_loader,          # 路由统计用的小批量数据
    calibration_forward_fn=lambda batch: model(**batch),
    tune_router=True,              # 同时微调 router/gate
    lora=True, lora_r=8, lora_alpha=16,   # 专家内注入 LoRA（进一步压参数量）
)
# 方式 B：手动指定专家（无需校准数据）
# config = EFSTConfig(expert_indices={"mlp.experts": [0, 3, 7]}, lora=False)

info = apply_efst(model, config)
print(info.summary())

# 之后用 8bit 优化器（只给可训练参数分配状态）
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

**说明**：冻结参数不产生梯度、不占优化器状态。支持 `lora=True`（经典专家注入 LoRA）
与 **3D 张量专家**（transformers 5.15+ Qwen3Next/Qwen3.5，`lora=True` 自动回退为按行解冻）。
实测（Qwen3Next 3 层混合，top-2）：可训练 725,712 → 143,600（19.8%），60 步 loss 下降 23%。
标准 ModuleList 结构（8 专家，top_k=3）实测：识别 8 专家、冻结 5 个、可训练 34856→4872（14%）。
> 注意：专家容器应为**标准 MoE 结构**（如 `mlp.experts` = 各专家子模块的 ModuleList）；
> 若用嵌套 `gate`+`experts` 的自定义 block，EFST 会把 `gate`+`experts` 当 2 个专家而非展开子专家。

### 4.8 核显（DirectML）块级常驻执行器（实验性，默认关闭）

> 仅适用于 Windows + `pip install torch_directml` 的环境；未检测到 DirectML 核显、
> 或回退条件不满足时，`--igpu` 会打印原因并自动回退纯 CPU，不影响正常训练。

```bat
:: 启用核显块级常驻执行器（实验性，默认关闭）
:: 注意：主训练环境（torch ≥2.6）不装 torch-directml（两者钉死的 torch 版本互斥，
:: 同环境安装会互相破坏）；跑 --igpu 请用独立 DML venv（.venv_dml，torch 2.4.1 +
:: torch_directml）。主环境直接跑 --igpu 会得到 venv 引导提示并自动回退纯 CPU。
D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe train.py --igpu

:: 自定义每步 token 上限 与 fp32 基座内存上限（MB）
D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe train.py --igpu --igpu_max_tokens 128 --igpu_mem_mb 4096

:: 调低核显启用门槛（校准比 ≥ 该值才启用）；0 = 跳过启动校准
D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe train.py --igpu --igpu_min_gain 1.05
```

**作用**：训练时把**全部权重常驻核显**（冻结基座只搬一次），前向+反向全程在核显上
执行，**每步只同步一次**——这是本机实测唯一有正收益的核显形态（GEMM 段约 +30%）。

**为什么不是「逐算子调度」**：`torch_directml` 的每次 `.to("cpu")` / `item()` 都是
一次**整队列排空（实测单次约 10~15ms）**，逐算子/逐层往返等于每步付 10+ 次排空，
实测比纯 CPU 慢约 2 倍。因此必须「权重常驻 + 整步不落地 + 每步一次同步」；可训练
参数（LoRA 适配器等）的梯度与更新后权重以小体积 CPU↔核显往返。

**自动回退条件**（任一满足即回退纯 CPU 并打印原因）：

| 条件 | 默认值 | 覆盖方式 |
|---|---|---|
| 每步 token 数（batch×seq） | ≤ 256 | `--igpu_max_tokens` / env `GPU_SCHED_MAX_TOKENS` |
| fp32 基座参数内存 | 按整机内存自适应：16GB→7000MB、12GB→5250MB、8GB→3500MB | `--igpu_mem_mb` / env `GPU_SCHED_MEM_MB` |
| 启动时 GEMM 校准核显/CPU 吞吐比 | ≥ 1.10 | `--igpu_min_gain` / env `GPU_SCHED_CALIB_MIN`（`=0` 跳过校准） |
| 模型含非 fp32 参数（量化基座） | 不支持 | — |
| 与 `--flash`（disk_balancer）同用 | 不兼容 | —（本次忽略 `--igpu`） |

**重要说明（实测结论，R5-4500U 同机）**：大 GEMM 流水线（50 个 2048²，末尾同步
一次）DML 294.5 GFLOPS vs CPU 226.7 GFLOPS = **1.30×**；真实 Qwen 结构全链
（8 层 1024 宽，权重常驻）**seq=128：1.35×**；但 **seq=512：0.70×**——DML 的
`F.sdpa` 在长序列上比 CPU 慢 3~7 倍（S=1024 实测 158ms vs 22.9ms），且专用
`multi_head_attention` 内核需要较新的 GPU 驱动（本机 27.20.11032 不支持）。
**结论：只建议 seq≤256 的短序列训练开启；长序列、小模型（H<1024）、量化基座
请保持纯 CPU。**

**兼容性与换机自查**：本执行器基于通用 DirectML（D3D12）接口，已在 AMD Radeon(TM)
Graphics（R5-4500U）实测；Intel UHD 630（i5-10400 等）走同一接口——自 R8 起执行器
为**换机自适应，无需人工预判**：内存上限按整机内存等比缩放，启动时 `calibrate_gpu()`
实测本机核显/CPU GEMM 吞吐比（两轮取较好、长预热促核显爬频——单轮实测有 ±0.1 抖动），
低于门槛（默认 1.10，`--igpu_min_gain 0` 跳过）自动回退并打印实测值，即「弱核显机器
开 --igpu 无害，只是自动不生效」。换机后运行 `py -3.11 gpu_scheduler.py`，末行直接
打印校准值与判定。若提示「Basic Render Driver / 软件适配器」，说明未安装或未启用
厂商图形驱动，执行器会自动拒绝启用。

**异步/并发压榨已证死路（R9）**：CPU/核显真并发技术上是可行的（后台排空线程，
混合负载 1.32×），但用于训练实测 0.83× vs 常驻——本机核显快于 CPU，分批给 CPU
只会坐上关键路径。不要尝试「半 CPU 半核显」的数据并行或算子切分方案。

**生图（SD）训练不要开核显**：同机实测 SD UNet 主力算子在 DML 上全面落后
（conv 主力形状 0.50~0.90×、S=1024 自注意力 0.14×、投影 GEMM 0.38~0.65×），
且 torch-directml 钉死 torch 2.4.1 与 diffusers 0.40 导入不兼容（修复：
`torch241_compat.py`，train_sd_lora.py 已在 import diffusers 前引入），真实 UNet
的 DML 反向还会触发插件崩溃。SD 训练保持纯 CPU（fp32）。

**Python 接口**（脚本内自行驱动，`train.py` 已内置同款流程）：

```python
from gpu_scheduler import IgpuExecutor

ex = IgpuExecutor(model)                 # 不满足条件时 prepare() 返回原因
print(ex.prepare(tokens=batch_size * seq_len))  # "OK" = 已启用，模型常驻核显
# 之后每步：model(input_ids=..., labels=...) -> loss.backward()
#          -> ex.grad_to_cpu() -> CPU 优化器 step -> ex.weights_from_cpu()
```

低阶接口 `big_gemm / big_linear / big_conv2d / patch_igpu` 仍保留（单算子实验用途），
但**逐算子调度在训练中为负收益，官方不推荐**。

---

## 5. 硬盘均衡负载（disk_balancer，SSD 保护）

> ### 免责声明与使用限制（请务必阅读）
> 本模块（`disk_balancer`，下称"本功能"）在**受支持的环境下**提供训练期内存均衡卸载
> 能力。**本功能并非对所有运行环境安全**。在**不受支持的环境**中使用本功能，**存在
> 导致存储介质（含宿主机文件系统）数据损坏、文件丢失、不可恢复性损伤的现实风险**，
> 该等风险由使用者自行承担。
>
> **受支持环境（仅限）**：
> - Windows（`os.name == "nt"`）——通过原生 ctypes 访问磁盘；
> - 非 WSL 的 Linux（真正的 ext4 等本地分区）。
>
> **明确不受支持、禁止使用**：
> - **WSL（Windows Subsystem for Linux）及其派生环境**。WSL 下宿主磁盘以 **9P 协议**
>   （网络文件系统语义）挂载于 `/mnt/c`、`/mnt/d` 等。在 WSL 中使用本功能，本功能将对
>   宿主 NTFS 进行高频读写/删除，**已实测可导致宿主 NTFS 主文件表（MFT）严重扰动、
>   数据无法落盘、文件丢失**（真实事故案例）。**任何在 WSL 中使用本功能造成的
>   数据损失，本项目不承担任何责任**。代码已在检测到 WSL 时自动禁用本功能
>   （`update_step` 空转、忽略 `--flash` 参数），但此防错机制不构成对使用者
>   在非支持环境下使用的授权或保证。
> - **系统盘剩余空间严重不足**、**训练期间强制关机/断电**等场景。
>
> **数据损失处置**：若已发生文件丢失，请**仅**以只读方式运行 `chkdsk <盘>:`（不带
> `/f`）进行诊断；**严禁使用 `chkdsk /f`、格式化或任何可能覆写受损介质的操作**，
> 上述操作可能将可恢复数据标记为已丢失。**建议在使用任何数据恢复操作前咨询专业人员。**
>
> **免责声明**：在获得明确授权前，请在**受支持环境**（如上）使用本功能。因在
> 非受支持环境（尤其 WSL）使用本功能所产生的任何数据损坏、丢失、系统故障或
> 其他损失，本项目（及其维护者）**不承担任何明示或暗示的责任**。

**作用**：当内存紧张时，自动将**不参与训练的冻结层权重（冷参数）**卸载至硬盘，释放内存，避免 Windows 虚拟内存导致 SSD 频繁擦写（训练期间磁盘活动 100% 会损害 SSD 寿命）。

本项目的 bitsandbytes CPU 后端在训练时提供**硬盘均衡负载**技术，可通过 `--flash` 参数激活（技术详情见 `docs_cpu/TECHNICAL_GUIDE.md` 与 `docs_cpu/TECH_REPORT.md`）。

> **注意事项**：该脚本在理想状态下可在内存紧张时继续训练，但**无法完全替代 Windows 虚拟内存**。训练过程中我们**仍建议用户保留至少 1GB 的虚拟内存**，以应对计划外的内存突发情况。

### 5.1 命令行控制

在训练命令末尾追加相应参数：

```bat
:: 自动模式（推荐）：内存使用率超过阈值时，将冷参数卸载至硬盘
py -3.11 train.py --flash auto

:: 速度优先：优先训练速度，不优先保护磁盘
py -3.11 train.py --flash auto --flash_speed

:: 手动模式：指定各磁盘缓存容量(MB)
py -3.11 train.py --flash manual --flash_sizes C:2048,D:4096

:: 指定缓存路径，并在训练结束后保留缓存文件
py -3.11 train.py --flash auto --flash_paths D:\cache,E:\cache --flash_keep

:: 调低阈值：内存使用率达到 60% 即触发卸载
py -3.11 train.py --flash auto --flash_threshold 0.6
```

### 5.2 参数表

| 参数 | 取值 | 说明 | 默认值 |
|---|---|---|---|
| `--flash` | `auto` / `a` / `true` = 自动；`manual` = 手动 | 启用硬盘均衡负载 | 关闭 |
| `--flash_sizes` | `C:2048,D:4096` | 手动模式下各磁盘缓存容量(MB) | 无 |
| `--flash_paths` | `D:\cache,E:\cache` | 缓存路径（逗号分隔） | 脚本所在目录 |
| `--flash_speed` | 布尔 | 速度优先模式（不优先保护磁盘） | `False` |
| `--flash_keep` | 布尔 | 训练结束后保留缓存文件 | `False` |
| `--flash_threshold` | 0.0~1.0 | 触发卸载的内存使用率阈值 | 0.8 |

### 5.3 冷参数读回（Python 接口）

```python
from disk_balancer import DiskLoadBalancer, DiskBalancerConfig
balancer = DiskLoadBalancer(DiskBalancerConfig(mode="auto"))
balancer.attach_model(model)      # 注册模型，自动检测冻结层冷参数
balancer.start()
# 训练循环内调用 balancer.update_step()  每步检查内存并自动迁移
t = balancer.get_cold("model.fc1.weight")   # 从磁盘读回冷参数
balancer.cleanup()                # 训练结束后删除缓存
```

### 5.4 DiskLoadBalancer 状态控制接口

| 方法 | 说明 |
|---|---|
| `attach_model(model)` | 注册模型，自动检测冻结层冷参数 |
| `start(script_dir=)` | 启动（自动分配缓存目录） |
| `update_step()` | 每一步检查内存，超过阈值时自动迁移一个冷参数（返回迁移数量） |
| `put_cold(key, tensor)` | 手动将某参数标记为冷参数并卸载至磁盘 |
| `get_cold(key)` | 从磁盘读回冷参数（mmap 零拷贝） |
| `get_cold_shape(key, shape)` | 按指定形状读回冷参数 |
| `contains(key)` | 判断指定键是否位于冷参数缓存中 |
| `drop(key)` | 删除指定冷参数缓存 |
| `stats()` | 返回迁移数量、冷参数数量、内存使用率、磁盘分布 |
| `wait_writes()` | 等待异步写盘完成 |
| `cleanup()` | 删除全部缓存文件 |
| `__enter__/__exit__` | 支持上下文管理器（退出时自动清理） |

### 5.5 在 Python 脚本中使用

> `--flash` 是 **CLI 参数**，仅对使用了 `add_flash_args()` 的命令行入口（如 `train.py`）
> 生效。若您在自己的 Python 脚本中调用，请使用以下两种方式。

**方式一：复用 `--flash` 参数（若您的脚本用 argparse）**

```python
import argparse
from disk_balancer import add_flash_args, parse_flash_args

parser = argparse.ArgumentParser()
add_flash_args(parser)              # 为脚本注入 --flash 等相关参数
args = parser.parse_args()

cfg = parse_flash_args(args)        # 若未传 --flash，返回 None（不启用）
if cfg is not None:
    balancer = DiskLoadBalancer(cfg)
    balancer.attach_model(model)
    balancer.start()
```

**方式二：直接构造配置对象（不经过命令行）**

```python
from disk_balancer import DiskBalancerConfig, DiskLoadBalancer

cfg = DiskBalancerConfig(
    mode="auto",              # "auto" / "manual"
    sizes={"C:": 2048, "D:": 4096},   # 手动模式各盘缓存(MB)
    paths=["D:\\cache", "E:\\cache"], # 缓存路径（缺省=脚本目录下 .flash_cache）
    speed_first=False,        # True=速度优先（不优先保护磁盘）
    keep_cache=False,         # True=训练后保留缓存文件
    memory_threshold=0.8,     # 内存使用率超过该值触发卸载
    min_param_size=100000,    # 参数元素数低于此值不卸载
    throttle_threshold=0.85,  # 磁盘活动率超过此值暂停写入
    min_free_ratio=0.05,      # 磁盘剩余空间低于此比例禁止写入
    offload_prefix="",        # 仅卸载前缀匹配的冷参数（空=全部）
)
balancer = DiskLoadBalancer(cfg)
balancer.attach_model(model)
balancer.start()
# 训练循环中每步调用 balancer.update_step() ...
```

**说明**：若未启用 `--flash`（或未构造 config），`cfg` 与 `balancer` 为 `None`，
`update_step()` 将直接返回 `0`，不影响原训练流程。

---

## 6. 开始使用

1. **首次**：安装 Python 与 PyTorch，完成「2. 编译」并确认 `selftest PASSED`。
2. **初步验证**：使用「4.1 量化冻结层」与「4.2 8-bit 优化器」执行一次小规模 LoRA 训练。
3. **运行监控**：后台启用「4.5 后台内存监视」，确认无磁盘异常读写。
4. **异常排查**：参考「7. 常见问题」。

---

## 7. 常见问题

| 现象 | 处理方法 |
|---|---|
| `selftest` 输出非 PASSED | 编译未成功：重新执行 `build_manual\build_manual.bat amd` |
| `No module named bitsandbytes` | `PYTHONPATH` 配置错误：检查「3. 环境配置」 |
| 训练期间磁盘频繁读写、速度缓慢 | 内存溢出导致交换：降低 `max_length` / `batch`，或启用「4.1 量化冻结层」 |
| 编译提示「未声明标识符」 | cpp 文件行尾格式错误：改用 CRLF，或编译时增加 `/utf-8` 参数 |
| 某一矩阵乘法长时间无响应 | 误用了 bf16：本环境需使用 fp32 |
| 加了 `--igpu` 训练反而更慢 | 通常为长序列（seq>256）或小模型：执行器会打印原因并自动回退纯 CPU；可用 `--igpu_max_tokens` 调整启用区间 |

---

## 8. 进阶文档

- 技术文档：`docs_cpu/TECHNICAL_GUIDE.md`（各内核的改动细节）
- 技术报告：`docs_cpu/TECH_REPORT.md`（完整实验数据与结论）
