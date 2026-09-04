# bitsandbytes CPU 爆改版 — 构建与使用指南（无 CUDA 环境）

> 目标机器：**无 NVIDIA 显卡**的 Windows 10 / Linux，AVX2 及以上 CPU
> （i5-10400 / R5-4500U 均满足）。整个构建链路**不需要任何 CUDA 组件**。

---

## 1. 这份改动提供了什么

| 能力 | 说明 | 实测效果 |
|---|---|---|
| **GDN 融合内核** `gdn_cpu` | Gated DeltaNet（Qwen3-Next / Qwen3.5 线性注意力层）前向+反向的 AVX2/FMA 融合内核，OpenMP 按 head 并行，反向用分块检查点 | 反向从 **728 s → 秒级**；fwd+bwd 比朴素 eager 循环快 **4~45×**；检查点内存 O(T)→O(T/C) |
| **8-bit 优化器融合内核** | kOptimizerStatic8bit 的 CPU 移植：dequant→更新→参数写回→requant 单遍融合，7 种优化器，标量/AVX2 双路径逐位一致 | 与 fp32 AdamW 终态 loss 差 < 0.01；状态体积 fp32 的 1/4 |
| **量化内核全家桶** | blockwise 8-bit quantize/dequantize（fp32/bf16/fp16）、fp4/nf4 4-bit dequant、int8 向量量化、4-bit GEMV 推理 | 全部带 CUDA 参考逐位对齐验证 |

层叠效果（T=1024, B=2, H=4, 64 维，单层 fwd+bwd）：
```
fused   36 ms   峰值内存 +42 MB
naive  1633 ms  峰值内存 +399 MB      （原 eager 路径）
```

## 2. 拉到你电脑上

改动共 **9 个文件**（对上游 bitsandbytes 仓库）：

```
bitsandbytes/
├── CMakeLists.txt                        # 改：加入 cpu_gdn.cpp、AVX2/ffp-contract 编译选项
├── bitsandbytes/
│   ├── gdn_cpu.py                        # 新：GDN Python 包装（本文 §5.1）
│   └── backends/cpu/ops.py               # 改：8-bit 优化器走融合 C 内核
├── csrc/
│   ├── cpu_gdn.cpp                       # 新：GDN 前向/反向 C++ 内核
│   ├── cpu_ops.cpp                       # 改：8-bit 优化器 + LUT 加速内核
│   ├── cpu_ops.h                         # 改：内核声明（含 layout 参数）
│   └── pythonInterface.cpp               # 改：ctypes C 入口
└── tests/test_cpu_e2e.py                 # 新：端到端验证
```

**两种方式任选：**

- **方式 A（推荐）**：把整个 `cpu_bundle.zip` 解压，其中 `bitsandbytes/` 文件夹
  按**相同相对路径**覆盖到你 clone 的上游 bitsandbytes 仓库上
  （`git clone https://github.com/bitsandbytes-foundation/bitsandbytes`）。
- **方式 B**：不 clone 上游，直接用 zip 里附带的改动文件 + 上游同版本源码自行拼接。

> 注意：zip 里的 `libbitsandbytes_cpu.so` 是 Linux 产物，Windows 上必须本地重编（下一节）。

## 3. Windows 10 编译（零 CUDA）

### 前置（只装这三样，没有任何 CUDA 依赖）

1. **Visual Studio 2022 生成工具**（勾选 "使用 C++ 的桌面开发"）
2. **CMake ≥ 3.22**（VS 安装器里勾，或官网装；≥ 3.30 更好，见下方说明）
3. **Python 3.9+**（你训练用的那个）

### 编译命令

在 **x64 Native Tools Command Prompt for VS 2022**（或已 `vcvars64` 的终端）里：

```bat
cd bitsandbytes
cmake -B build -DCOMPUTE_BACKEND=cpu
cmake --build build --config Release
```

产物自动落到 `bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll`（Windows）或
`bitsandbytes\libbitsandbytes_cpu.so`（Linux）；CMake 已配置好输出目录。

### 3.1 手动 cl 直编（CMake 卡住时的可靠替代）

实测 CMake 4.3.1（VS2026 自带）+ 本机环境有两个坑：

1. 旧 `build/` 目录里若残留**其他环境**生成的 `CMakeCache.txt`（例如 WSL
   `/workspace/bitsandbytes` 的缓存），configure 直接报
   "CMakeCache.txt directory is different than ..." —— 删掉 `build/` 重来；
2. 全新 configure 也可能**卡死在 "Detecting CXX compiler ABI info"**
   （TryCompile 目录生成了 .obj/.exe 但 cmake 一直不返回）。

绕开 CMake，直接用 cl 编译（只 3 个 cpp，产物一致，AVX2 选项完全一样）：

```bat
:: 在 x64 Native Tools Command Prompt for VS 2022/2026 里
cd bitsandbytes
build_manual.bat
```

`build_manual.bat` 内容等价于：

```bat
cl /nologo /O2 /Ob2 /arch:AVX2 /fp:fast /openmp:experimental /std:c++17 /EHsc ^
   /DNOMINMAX /DNDEBUG /DWIN32 /D_WINDOWS /I csrc /LD ^
   csrc\cpu_ops.cpp csrc\cpu_gdn.cpp csrc\pythonInterface.cpp ^
   /Fe:bitsandbytes\libbitsandbytes_cpu.dll /link /DEF:build_manual\export.def
```

要点：

- **`/openmp:experimental` 必须加**：`cpu_ops.cpp` 用了 `#pragma omp simd`，
  普通 `/openmp` 会报 C7660；
- **`.def` 导出不可省**：`extern "C"` 符号在 MSVC 下默认不导出 DLL 符号
  （CMake 靠 `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` 兜底），`export.def` 列出
  GDN 内核 + 全部 `c*` CPU 入口；
- 产物 `libbitsandbytes_cpu.dll` 依赖 `VCOMP140.DLL`（MSVC OpenMP 运行时），
  Win10/11 装了 VC++ 2015-2022 Redist 即自带（`C:\Windows\System32`）；
- 验证：`python -m bitsandbytes.gdn_cpu` 全绿 + `python tests\test_cpu_e2e.py` 全过。

### Win10 已知的坑（都已替你处理，列出来以防万一）

| 坑 | 状态 |
|---|---|
| Windows SDK 的 `min/max` 宏与 `std::min/std::max` 冲突 | CMake 全局加 `NOMINMAX` |
| MSVC 不吃 GCC 的 `-mavx2` 写法 | CMake 对 x64 自动 `/arch:AVX2` |
| AVX2 机器上误跑 AVX512 指令 | 编译目标只有 AVX2+FMA；AVX512 路径仅编译期探测可用才开 |
| MSVC OpenMP 没有 persistent thread pool | `OpenMP_RUNTIME_MSVC=experimental`（CMake ≥ 3.30 生效，老版本自动退回 vcomp，**功能正确只是线程池每次重建**，不影响结果） |
| 符号导出 | `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON` |

**CPU 要求**：AVX2 + FMA（i5-10400 和 R5-4500U 都是 Comet Lake / Zen2，均原生支持）。
不支持 AVX2 的老 CPU 无法运行（内核编译目标即 AVX2）。

## 4. Python 侧启用

```bat
:: 方式一：临时（加环境变量）
set PYTHONPATH=C:\path\to\bitsandbytes
python train.py

:: 方式二：装进环境（editable）
pip install -e C:\path\to\bitsandbytes
```

验证安装（应全绿）：

```bat
python -m bitsandbytes.gdn_cpu          :: GDN 前向/反向精度自检（3 dtype）
python tests\test_cpu_e2e.py            :: 8-bit 优化器端到端收敛 + GDN 训练
```

## 5. 新增 API 参考

### 5.1 GDN：`bitsandbytes.gdn_cpu`

#### 直接调用

```python
from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule

o, final_state = fused_recurrent_gated_delta_rule(
    q,                # [B,T,H,K]  fp32/bf16/fp16，已含 1/sqrt(K) 缩放的话直接传入
    k,                # [B,T,H,K]  与 q 同 dtype
    v,                # [B,T,H,V]  与 q 同 dtype
    beta=None,        # [B,T,H]    可省略（默认全 1）
    g=None,           # [B,T,H]    每步 log 衰减，可省略（默认无衰减）
    scale=None,       # float      乘在 q 上的缩放（fla 语义）
    initial_state=None,     # [B,H,K,V] 初始状态
    output_final_state=False,  # True 才返回 final_state（省一次 dtype 转换）
    use_qk_l2norm_in_kernel=False,  # True 则在核外用 torch 做 l2 归一化（反向精确）
    head_first=False, # True 接受/返回旧版 fla [B,H,T,D] 布局
)
```

要点：
- `r=q` 是 `q` 的别名（fla ≥ 0.3 把首参改名 `r`，两种调用都支持）；
- **连续输入零拷贝**：`[B,T,H,D]` 连续张量直接原地读；跨步输入自动回退到
  逐 head 连续拷贝（结果逐位一致，只是多一次拷贝）；
- 三种 dtype（fp32/bf16/fp16）内部全 fp32 累加，bf16/fp16 只在出入核转换；
- 任意线程数输出**逐位一致**（head 间无共享状态）。

#### 训练（autograd 自动生效）

输入 `requires_grad_(True)` 后正常 `loss.backward()` 即可，dq/dk/dv/dbeta/dg/ds0
全部解析计算，梯度 dtype 自动匹配输入 dtype（bf16 模型输入 → bf16 梯度）。

#### 一键接入 Qwen3-Next / Qwen3.5（transformers）

```python
from bitsandbytes.gdn_cpu import patch_transformers
patch_transformers()          # 之后 model(input_ids) 的 GDN 层自动走 CPU 融合内核
```

#### 一键接入 fla（如装了 flash-linear-attention）

```python
from bitsandbytes.gdn_cpu import patch_fla
patch_fla()                   # CPU 张量走融合内核，CUDA 张量仍走原 fla
```

#### 环境变量

| 变量 | 作用 |
|---|---|
| `GDN_CPU_CHUNK` | 钉死反向检查点分块长度 C（正整数）；**不设则自适应**：按可用内存预算（25%，夹在 [512MB, 4GB]）取最大 C≤512，最小化跨层存活的检查点总量 |
| `GDN_CPU_LIB` | 显式指定动态库路径（找不到时用） |
| `GDN_CPU_DISABLE=1` | 强制抛错，用于验证回退路径 |
| `BNB_CPU_NO_AVX2=1` | 强制标量路径（A/B 调试用） |

### 5.2 8-bit 优化器（用法与 GPU 完全相同）

```python
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)   # 其余照旧
```

CPU 后端已注册的融合内核覆盖：**adam / lamb, momentum / lars, lion, rmsprop,
adagrad, ademamix**。调用 `opt.step()` 时自动走
`bitsandbytes::optimizer_update_8bit_blockwise` 的 CPU 实现
（dequant → 更新 → requant 单遍融合）；遇到不支持的组合自动回退到
PyTorch 组合路径，行为不变。

特性：
- 标量与 AVX2 路径**逐位一致**（含量化 LUT 加速路径，与 CUDA 参考对齐）；
- 非连续梯度自动回退安全路径（不会静默读错内存）；
- NaN/Inf 梯度语义与 CUDA 一致（状态清零、参数不动）。

### 5.3 量化内核（一般经由 `bnb.nn.Linear4bit/8bit` 间接使用）

CPU 注册的 torch 算子：`quantize_blockwise` / `dequantize_blockwise`（fp32/bf16/fp16）、
`dequantize_4bit`（fp4/nf4）、`int8_linear_matmul`、`gemv_4bit`。
无需手动调用；`Linear4bit`/`Linear8bitLt` 在 CPU 张量上自动分派到这些实现。

## 6. 验证命令清单

```bat
python -m bitsandbytes.gdn_cpu            :: GDN 精度自检
python tests\test_cpu_e2e.py              :: 优化器收敛 + GDN 训练 + 非连续梯度
python stress_opt.py                      :: 7 优化器 × 10 尺寸 × 3 dtype 压力 + 确定性
python gdn_layout_probe.py                :: 布局零拷贝 / C 不变性 / 峰值内存
python gdn_mem_probe.py                   :: 多层场景内存（T=8192 × 8 层）
python gdn_train_smoke.py                 :: fused vs naive 速度内存对比 + 可训练性
```

## 7. 内存调优（免得拷打 SSD）

**GDN 检查点自适应（新）**：反向需要的分块检查点从"每层 forward 存活到该层
backward"，训练时**所有层共存**，是长序列内存的大头。自适应策略在
"检查点总量（随 C 增大而减）"与"单层 backward 重算缓冲（随 C 增大而增，
仅瞬时存在）"之间取最优——检查点是持久项，所以取预算内最大 C：

| 场景（T=8192, H=16, K=V=128） | 旧固定 C=64 | 自适应（C=512） |
|---|---|---|
| 8 层 forward 峰值 | 2077 MB | **1181 MB（-900 MB）** |
| 8 层检查点总量 | 1074 MB | **134 MB（8×）** |
| 速度 | 2090 ms | 2023 ms（持平） |

结果与 C=64 **逐位一致**（C 不变性探针验证）。层数越多省得越多：12 层的
Qwen3.5-0.8B 约省 1.3 GB。

**12/16GB 内存建议**：
- 12GB（i5-10400）：batch×T 控制在 ~24k token 内；自适应 C 即可，别手动钉小 C；
- 16GB（R5-4500U）：可到 ~32k token；仍紧张就 `GDN_CPU_CHUNK=256` 折中；
- 优化器一律用 8-bit 版（状态直接砍到 1/4，这是最便宜的大头）；
- Windows 页面文件：**固定大小**（初始=最大，比如 8192-16384MB）且尽量放
  非 SSD 盘；放 SSD 也务必固定大小，避免运行时动态扩容反复擦写。

## 8. 故障排查

| 症状 | 处理 |
|---|---|
| `gdn_cpu native library not found` | 先跑 §3 编译；或 `set GDN_CPU_LIB=C:\...\libbitsandbytes_cpu.dll` |
| `GDN backward called without checkpoints` | forward 是在 `torch.no_grad()` 里跑的；训练时保持梯度开启 |
| CPU 占用只有一半 | 正常：并行度上限是 B×H（head 数）；也检查 `OMP_NUM_THREADS` 是否被设成逻辑核数（应=物理核数，用 torch_cpu_kit 自动处理） |
| 数值可疑 | `set BNB_CPU_NO_AVX2=1` 切标量路径对比；逐位应一致 |
