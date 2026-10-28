"""Update bitsandbytes/docs_cpu (CN+EN) with the three workstreams that were missing.

Missing before this patch:
  * 核显 OpenCL/CLBlast path -- the docs only had the Windows DirectML executor (QUICKSTART 4.8 /
    TECH_GUIDE 3.8 / TECH_REPORT 3.7). The CLBlast work is a different backend with its own numbers.
  * 4-bit optimizer (AdamW4bit) -- only the generated API reference mentioned it.
  * 2-bit / 3-bit quantisation (quant2bit.py, quantnbit.py) -- absent everywhere.
  * the guide's "relative to v0.45.1" claim, which is stale (upstream base is 0.50.2/main).

Numbers are taken from the measured record: report sections 7.104-7.110 (CLBlast / fp16 / resident
step / overlap), 7.76 / 7.80 / 7.107 (4-bit optimizer), and the 2/3-bit modules' own docstrings plus the
SD1.5 end-to-end verdict.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DOCS = r'D:\work\bitsandbytes-CPU\bitsandbytes\docs_cpu'

QUICKSTART_ZH = '''### 4.11 核显加速（OpenCL / CLBlast，实验性）

§4.8 的核显路径是 Windows 专用的 DirectML 执行器。另有一条**跨平台**路径：OpenCL + CLBlast，
用于回答"这块核显到底能算多快、算得对不对"。

**调用约定（踩过的坑）**：CLBlast 的 C API 是**列主序**的。用行主序思维传非立方矩阵会返回
`-1011/-1015/-1016`（参数校验失败，而不是内核崩溃）。正确做法是"列主序调用 + 转置映射"，
零拷贝：

```
C_row(m×n) = A_row(m×k) @ B_row(k×n)
  ⇔ Sgemm(COL, NO, NO, m=n, n=m, k=k,
          A=B_row 在显存里的地址, lda=n,
          B=A_row 在显存里的地址, ldb=k,
          C=C_row 在显存里的地址, ldc=n)
```

**实测（Vega 6 / R5-4500U，fp32，CLBlast）**：

| 形状 | 核显 GFLOPS | 同形状 CPU |
|---|---|---|
| MLP up (k=2048→n=2048) | 535.0 | 217 |
| MLP down (k=2048→n=512) | 512.5 | 256 |
| QKV (k=512→n=2048) | 531.5 | 235 |
| o proj (k=512→n=512) | 371.5 | 164 |
| head (k=512→n=1024) | 486.0 | 219 |
| attn QK / AV | 116.7 / 119.5 | — |

套件平均 **505~551 GFLOPS vs CPU 204 ⇒ 2.6~2.7×**。fp16 纯运算（unroll ×4、16~24 累加器、
worksize 98304）实测 **1290 GFLOPS = GT 1030 理论值 1127 的 1.14×**；但 fp16 GEMM 最高 1039
（0.92×），fp32 GEMM 655（0.58×）。

**设备常驻单层训练步**（权重常驻显存、loss/反向/Adam 全在设备上，host 每步只见 4 字节 loss）：

| 层宽 | CPU | 核显常驻 | 加速 |
|---|---|---|---|
| 2048 | 202.9 ms | 135.1 ms | **1.50×** |
| 1024 | — | — | 1.35× |
| 512 | — | — | **CPU 反而快 1.24×** |

权重对拍相对误差 7.6e-05，先对拍再读速度。

**什么时候用核显（由 M 决定，不是"谁快"）**：M=1 给 CPU；**M≥64 给核显（约 2×）**。
**异步并发不要做**：CPU 与核显同时跑时，iGPU 只保留 40~54%、CPU 只剩 53~61%（共享的 8 MB L3
被 CPU 工作集挤掉），并发 1.1~1.3× 反而不如顺序执行 1.51×。

**复跑脚本**（仓库根目录，需要 `pyopencl` + CLBlast 动态库）：

```bat
py -3.11 clblast_colmajor_shapes.py     :: CLBlast 列主序修复 + 真实形状 GFLOPS 表
py -3.11 igpu_clblast_resident.py       :: 设备常驻单层训练步（先对拍、后计时）
py -3.11 igpu_fp16_tune.py              :: fp16 纯运算调参（nacc/worksize/unroll 网格）
```

⚠️ **状态：实验性，尚未接入训练/推理主路径**，也没有随 wheel 发布 Python API；把它当
"测量与验证工具"用。它与 §4.8 的 DirectML 执行器是两条不同的路（DirectML 仅 Windows，
OpenCL 跨平台），结论不能混用。

### 4.12 4-bit 优化器（AdamW4bit：状态再减半）

8-bit 优化器的两阶矩状态已压到约 2.3 B/参数；`AdamW4bit` 再把状态压到 **约 1.03 B/参数**
（fp32 AdamW 是 8.0）。用法与 8-bit 优化器同形：

```python
from bitsandbytes.optim import AdamW4bit

opt = AdamW4bit(model.parameters(), lr=1e-4, weight_decay=0.01)
opt.zero_grad()
loss = loss_fn(model(x), y)
loss.backward()
opt.step()               # 训练循环其余部分不变
```

**状态布局**：两阶矩各按 4-bit 码保存（每字节两个码，高半字节是偶数下标），每 256 个元素一个
fp32 absmax；16 个对称电平（码 0 = −1，码 15 = +1）。**精确 0 不在码表里**——全零块写入码 8
（对应 +1/15），因为 absmax 也是 0，反量化后仍是 0。

**收益（在 30.49M 参数的 AR 模型上实测量）**：

| 项 | 数值 |
|---|---|
| 状态内存 | 232.6 MB → **29.5 MB**（8-bit 优化器的 1/2，fp32 的 1/8） |
| 内核步耗时（1M 参数） | 0.658 ms vs 8-bit 1.682 ms = **2.56×** |
| 内核步耗时（≥64M 参数） | **1.74×**（5 次：1.69~1.81）——量级越大收益越小，别拿 1M 的数字外推 |
| 训练轨迹（120 步 AR） | 两条 loss 曲线相差 **0.27%**（不是"一致"，是"同轨"） |
| 状态码正确性 | 与独立 numpy 实现 50 步**逐位一致** |

**边界**（每条都是踩过的）：

- 只支持 **fp32 参数**；`blocksize` 编译期固定 256，传其它值直接报错
- 它是**独立类**，不替换 `torch.optim.AdamW`；量化状态**不保证可复现轨迹**（同 seed 同数据，路径与 fp32 不同）
- DLL 里没有 `coptimizer_update_4bit_blockwise_cpu` 时**导入即报错**，不会静默退回 8-bit
- 发布状态：`0.50.2.dev3` 起的构建提供

### 4.13 2-bit / 3-bit 量化（quant2bit / quantnbit）

**2-bit（`quant2bit.py`）**：块级（block=64）最近电平量化，**每字节打包 4 个索引**。
存储 **2.5 bit/权重**（2 + 32/64），对比 4-bit 路径的 4.5 bit 和 fp32 的 32 bit。

**3-bit（`quantnbit.py`）**：通用 N-bit 打包（LSB 优先，3-bit 会跨字节边界），自带
"打包→解包"自检。3-bit 选 **uniform** 电平而不是 NF：两者损失只差 0.0003（噪声内），
uniform 让解码回到一次乘加。

**损失代价**（训练好的 32×32 AR 模型，512 条验证样本）：

| 存储格式 | 验证 loss | 相对 fp32 |
|---|---|---|
| fp32 | 5.5038 | — |
| 4-bit（uniform / nf） | 5.5080 / 5.5084 | +0.004 |
| 3-bit（uniform / nf） | 5.5234 / 5.5231 | +0.019 |
| **2-bit nf** | **5.6206** | **+0.117（≈2.1%）** |
| 2-bit uniform | 5.6354 | +0.132 |

⇒ **2-bit 时 NF 比 uniform 好 11%；3-bit 及更宽时 NF 没有意义。**

**两个端到端判决**：

- ⚠️ **SD1.5 全 pipeline 用 2-bit 权重：不成立** —— 存储省 39.5%，但生成的图彻底坏掉
- ✓ **冻结的 2-bit 基座 + LoRA 训练：成立** —— 只替换 `nn.Linear`，初始验证 loss
  5.5538（与格式自身的值一致），LoRA 参数可训、loss 下降，基座不需要可微

```python
from quant2bit import quantize, dequantize, bytes_per_weight      # 2-bit：2.5 bit/权重
from quantnbit import quantize as quantize_n, dequantize as dequantize_n   # 任意 N-bit

packed, absmax, meta = quantize(w, block=64)        # 2-bit 打包（4 码/字节）
w2 = dequantize(packed, absmax, *meta)
print(bytes_per_weight(64))                         # 2.5

packed3, absmax3, meta3 = quantize_n(w, bits=3, block=64)   # 3-bit
w3 = dequantize_n(packed3, absmax3, meta3)
```

⚠️ **状态：脚本级组件（仓库根目录），尚未进 wheel**；它是"格式与测量"的参考实现，
不是训练主路径的一部分。

'''

QUICKSTART_EN = '''### 4.11 Integrated-GPU acceleration (OpenCL / CLBlast, experimental)

The iGPU path in 4.8 is the Windows-only DirectML executor. There is a second, **cross-platform**
path: OpenCL + CLBlast, built to answer "how fast is this iGPU really, and is the result correct".

**The call convention (the trap)**: CLBlast's C API is **column-major**. Passing non-cubic matrices
with a row-major mindset returns `-1011/-1015/-1016` (argument validation failures, not a crashing
kernel). The fix is a column-major call with a transpose mapping, zero copies:

```
C_row(m×n) = A_row(m×k) @ B_row(k×n)
  <=> Sgemm(COL, NO, NO, m=n, n=m, k=k,
            A=B_row buffer, lda=n,
            B=A_row buffer, ldb=k,
            C=C_row buffer, ldc=n)
```

**Measured (Vega 6 / R5-4500U, fp32, CLBlast)**:

| Shape | iGPU GFLOPS | CPU, same shape |
|---|---|---|
| MLP up (k=2048->n=2048) | 535.0 | 217 |
| MLP down (k=2048->n=512) | 512.5 | 256 |
| QKV (k=512->n=2048) | 531.5 | 235 |
| o proj (k=512->n=512) | 371.5 | 164 |
| head (k=512->n=1024) | 486.0 | 219 |
| attn QK / AV | 116.7 / 119.5 | — |

Suite average **505-551 GFLOPS vs CPU 204 => 2.6-2.7x**. fp16 pure compute (unroll x4, 16-24
accumulators, worksize 98304) measures **1290 GFLOPS = 1.14x the GT 1030's theoretical 1127**; but
fp16 GEMM peaks at 1039 (0.92x) and fp32 GEMM at 655 (0.58x).

**Device-resident single-layer training step** (weights resident on the device, loss/backward/Adam all
on-device, the host sees a 4-byte loss per step):

| Layer width | CPU | iGPU resident | Speedup |
|---|---|---|---|
| 2048 | 202.9 ms | 135.1 ms | **1.50x** |
| 1024 | — | — | 1.35x |
| 512 | — | — | **CPU wins at 1.24x** |

Weights agree to 7.6e-05 relative error -- correctness before speed.

**When to use the iGPU (it is decided by M, not by "which is faster")**: M=1 goes to the CPU;
**M>=64 goes to the iGPU (about 2x)**. **Do not run CPU and iGPU concurrently**: when both run, the
iGPU retains only 40-54% and the CPU 53-61% (the shared 8 MB L3 is thrashed by the CPU working set),
so concurrency gives 1.1-1.3x against 1.51x for sequential execution.

**Scripts to reproduce** (repo root; needs `pyopencl` and the CLBlast shared library):

```bat
py -3.11 clblast_colmajor_shapes.py     :: column-major fix + real-shape GFLOPS table
py -3.11 igpu_clblast_resident.py       :: device-resident single-layer step (check, then time)
py -3.11 igpu_fp16_tune.py              :: fp16 pure-compute tuning (nacc/worksize/unroll)
```

⚠️ **Status: experimental, not wired into the training/inference path** and not shipped as a Python
API in the wheel. Treat it as a measurement/verification tool. It is a different path from the
DirectML executor in 4.8 (DirectML is Windows-only, OpenCL is cross-platform); do not mix conclusions.

### 4.12 The 4-bit optimizer (AdamW4bit: half the state of 8-bit)

The 8-bit optimizer already stores its two moments at about 2.3 B/parameter; `AdamW4bit` takes the
state down to **about 1.03 B/parameter** (fp32 AdamW is 8.0). Its API has the same shape:

```python
from bitsandbytes.optim import AdamW4bit

opt = AdamW4bit(model.parameters(), lr=1e-4, weight_decay=0.01)
opt.zero_grad()
loss = loss_fn(model(x), y)
loss.backward()
opt.step()               # the rest of the training loop is unchanged
```

**State layout**: each moment is stored as 4-bit codes (two codes per byte, high nibble = even index)
with one fp32 absmax per 256 elements; 16 symmetric levels (code 0 = -1, code 15 = +1). **Exact zero
is not in the code table** -- an all-zero block writes code 8 (+1/15), whose dequantised value is
still 0 because its absmax is 0.

**Measured payoff (on the 30.49M-parameter AR model)**:

| Quantity | Value |
|---|---|
| State memory | 232.6 MB -> **29.5 MB** (half of the 8-bit optimizer, 1/8 of fp32) |
| Kernel step (1M params) | 0.658 ms vs 1.682 ms for 8-bit = **2.56x** |
| Kernel step (>=64M params) | **1.74x** (5 runs: 1.69-1.81) -- the larger the tensor, the smaller the gain; do not extrapolate the 1M number |
| Training trajectory (120 AR steps) | the two loss curves stay within **0.27%** (same track, not identical) |
| State codes | **bit-identical** to an independent numpy implementation over 50 steps |

**Limits** (each one encountered the hard way):

- fp32 parameters only; `blocksize` is compiled in at 256 and any other value raises
- it is a **separate class**, not a replacement for `torch.optim.AdamW`; quantised state does not
  promise reproducible runs (same seed and data still take a different path from fp32)
- if the DLL does not export `coptimizer_update_4bit_blockwise_cpu`, **import fails loudly** instead
  of falling back to 8-bit
- release status: available from the `0.50.2.dev3` build onward

### 4.13 2-bit and 3-bit quantisation (quant2bit / quantnbit)

**2-bit (`quant2bit.py`)**: block-wise (block=64) nearest-level quantisation, **four indices packed per
byte**. Storage is **2.5 bits/weight** (2 + 32/64), against 4.5 bits for the 4-bit path and 32 for fp32.

**3-bit (`quantnbit.py`)**: a general packed N-bit quantiser (LSB first; 3-bit crosses byte
boundaries) with a built-in pack/unpack self-test. 3-bit uses **uniform** levels rather than NF: the
difference is 0.0003 (inside noise), and uniform keeps decoding a multiply-add.

**Loss cost** (trained 32x32 AR checkpoint, 512-sample validation slice):

| Storage format | Validation loss | vs fp32 |
|---|---|---|
| fp32 | 5.5038 | — |
| 4-bit (uniform / nf) | 5.5080 / 5.5084 | +0.004 |
| 3-bit (uniform / nf) | 5.5234 / 5.5231 | +0.019 |
| **2-bit nf** | **5.6206** | **+0.117 (~2.1%)** |
| 2-bit uniform | 5.6354 | +0.132 |

=> **at 2 bits NF beats uniform by 11%; at 3 bits and wider NF buys nothing.**

**Two end-to-end verdicts**:

- ⚠️ **SD1.5 full pipeline with 2-bit weights: does not work** -- 39.5% storage saved, but the
  generated images are destroyed
- ✓ **Frozen 2-bit base + LoRA training: works** -- replace only `nn.Linear`; the initial validation
  loss is 5.5538 (matching the format's own number), LoRA parameters train and the loss drops, and the
  base never needs to be differentiable

```python
from quant2bit import quantize, dequantize, bytes_per_weight      # 2-bit: 2.5 bits/weight
from quantnbit import quantize as quantize_n, dequantize as dequantize_n   # any N-bit

packed, absmax, meta = quantize(w, block=64)        # 2-bit, 4 codes per byte
w2 = dequantize(packed, absmax, *meta)
print(bytes_per_weight(64))                         # 2.5

packed3, absmax3, meta3 = quantize_n(w, bits=3, block=64)   # 3-bit
w3 = dequantize_n(packed3, absmax3, meta3)
```

⚠️ **Status: script-level components (repo root), not in the wheel**; they are the reference
implementation of the format and its measurements, not part of the training path.

'''

GUIDE_ZH = '''### 3.9 核显后端（OpenCL/CLBlast）：调用约定、fp16 与设备常驻步

**为什么另起一条路**：DirectML 执行器（§3.8）只能在 Windows 上跑，且受驱动算子支持限制。
OpenCL + CLBlast 用来把"核显能算什么、算到多快"测量清楚——它不依赖 torch_directml。

**列主序调用约定（核心修复）**：CLBlast 的 C API 按列主序解释矩阵。行主序思维下，立方矩阵
碰巧能对（转置后仍立方），**非立方矩阵则返回参数校验错误**（-1011/-1015/-1016，不是内核崩溃）。
正确做法是把"行主序的 C=A@B"映射成一次列主序 Sgemm（零拷贝，转置只体现在 m/n 与 lda/ldb/ldc 上，
见 QUICKSTART §4.11 的映射式）。这解释了整整一轮"CLBlast 在真实形状上不能用"的误判。

**fp16 纯运算**：AMD 驱动下编译器不会为 `mad` 生成 packed f16 指令，但 ISA 里有
`v_pk_add_f16`。用 unroll ×4 + 16~24 个独立累加器 + worksize 98304 的 OpenCL 内核，实测
**1290 GFLOPS**（= GT 1030 理论值 1127 的 1.14×）。GEMM 另算：fp16 GEMM 最高 1039（0.92×），
fp32 GEMM 655（0.58×）——纯运算峰值不等于 GEMM 峰值。

**设备常驻训练步**：早期版本每步都在 host 与 device 之间来回（下载 y 47 ms、torch 里算 loss
16 ms、numpy 转置 32 ms、上传 4 ms），host 胶水吃掉收益。常驻版的形态是：y 留在设备缓冲、
一个小内核算 `d = y - target` 并同时写出梯度与其转置、一个工作组归约 `sum(d²)`；host 每步只见
4 字节 loss 和下一批数据。2048 宽层实测 CPU 202.9 ms → 设备 135.1 ms（1.50×），权重对拍 7.6e-05。

**为什么不能用"CPU+核显并发"**：两者共享 8 MB L3，CPU 工作集把核显的工作集挤出去 ⇒
iGPU 只剩 40~54%、CPU 剩 53~61%，并发 1.1~1.3× 不如顺序执行 1.51×。

**边界**：脚本级、未进 wheel；需要 `pyopencl` 与 CLBlast 动态库；M=1 归 CPU、M≥64 才归核显。

### 3.10 4-bit 优化器内核：块级 4-bit 状态 + AVX2 打包

**只换存储层**：8-bit 优化器的模板已把"AdamW 数学"与"状态存储"分开，4-bit 版本只替换存储
（`state1/state2` 每元素 4 bit，另加每 256 元素一个 fp32 absmax）。码表是 16 个对称电平，
码 0 = −1、码 15 = +1。

**两个实现细节**：① 打包必须用 `pshufb` 而不是饱和加法（`packus` 会把负数饱和掉，状态码
静默错）；② 舍入不调用库函数（call-free rounding），否则每个参数两次函数调用的开销会把
"少一半字节"的收益吃掉。标量参考实现与独立 numpy 版本 50 步逐位一致。

**速度的两个区间**：1M 参数时 0.658 ms vs 8-bit 1.682 ms（2.56×，内核比较隔离了状态宽度）；
参数规模 ≥64M 时收益降到 1.74×（5 次 1.69~1.81）——状态流量与参数/梯度流量的比例随规模变化，
说明"2.63×"只是 1M 参数的数字，不能外推。

### 3.11 NF2/NF3：N-bit 打包格式与"按测量选 level"

**打包**：n 个值每个 `bits` 位，从 LSB 开始连续写入 `ceil(n*bits/8)` 字节；值 k 从 bit
`k*bits` 开始，**3-bit 会跨字节边界**（2/4-bit 不会）。写错是静默的，所以模块自带
"打包已知索引序列→读回比对"的自检。

**电平选择由测量决定**：在 32×32 AR 模型上，2-bit 时 NF 比 uniform 好 11%（5.6206 vs
5.6354）；3-bit 时 NF 只领先 0.0003（噪声内）⇒ 3-bit 用 uniform，让解码保持一次乘加。
存储公式是 `bits + 32/block`：block=64 时 2-bit 为 2.5 bit/权重、3-bit 为 3.5 bit/权重。

**为什么不用现成的 8-bit 索引路径**：通用路径按 8-bit 索引查任意码表，**存储不省**；
要省存储必须把索引位宽本身降下来——这就是两个模块存在的理由。

'''

GUIDE_EN = '''### 3.9 iGPU backend (OpenCL/CLBlast): call convention, fp16, and the resident step

**Why a second path**: the DirectML executor (3.8) only runs on Windows and is limited by driver
operator coverage. OpenCL + CLBlast exists to measure what the iGPU can compute and how fast, without
depending on torch_directml.

**The column-major call convention (the core fix)**: CLBlast's C API interprets matrices column-major.
With a row-major mindset a cubic matrix accidentally works (transposing a cube leaves a cube), while
**non-cubic shapes return argument-validation errors** (-1011/-1015/-1016, not a crashing kernel). The
fix maps row-major "C = A @ B" onto one column-major Sgemm with zero copies -- only m/n and
lda/ldb/ldc change (the mapping is written out in QUICKSTART 4.11). This explains a whole round of
"CLBlast does not work on real shapes" that was really a wrong call.

**fp16 pure compute**: under the AMD driver the compiler does not emit packed f16 for `mad`, but the
ISA has `v_pk_add_f16`. An OpenCL kernel with unroll x4, 16-24 independent accumulators and worksize
98304 measures **1290 GFLOPS** (1.14x the GT 1030 theoretical 1127). GEMM is separate: fp16 GEMM peaks
at 1039 (0.92x), fp32 GEMM at 655 (0.58x) -- pure-compute peak is not GEMM peak.

**Device-resident training step**: the early version round-tripped per step (download y 47 ms, compute
the loss in torch 16 ms, transpose in numpy 32 ms, upload 4 ms) and the host glue ate the gain. The
resident version keeps y in its device buffer, one small kernel computes `d = y - target` and writes the
gradient together with its transpose, and one workgroup reduces `sum(d^2)`; the host sees a 4-byte loss
and the next batch. A 2048-wide layer measures CPU 202.9 ms -> device 135.1 ms (1.50x), weights agree
to 7.6e-05.

**Why CPU+iGPU concurrency does not work**: the two share the 8 MB L3, and the CPU working set evicts
the iGPU's, so the iGPU retains 40-54% and the CPU 53-61%; concurrency gives 1.1-1.3x against 1.51x
sequential.

**Limits**: script-level, not in the wheel; needs `pyopencl` and the CLBlast shared library; M=1 belongs
to the CPU, only M>=64 belongs to the iGPU.

### 3.10 The 4-bit optimizer kernel: block-wise 4-bit state + AVX2 packing

**Only the storage layer changes**: the 8-bit template already separates the AdamW math from the state
storage, so the 4-bit version replaces the storage only (`state1/state2` at 4 bits per element plus one
fp32 absmax per 256 elements). The code table is 16 symmetric levels, code 0 = -1, code 15 = +1.

**Two implementation details**: (1) packing must use `pshufb`, not saturating adds (`packus` saturates
negative values away and the state codes go silently wrong); (2) rounding is call-free -- two library
calls per parameter would eat the gain from moving half the bytes. The scalar reference is bit-identical
to an independent numpy implementation over 50 steps.

**Two speed regimes**: at 1M parameters 0.658 ms vs 1.682 ms for the 8-bit kernel (2.56x, the comparison
isolates state width); at >=64M parameters the gain drops to 1.74x (5 runs 1.69-1.81) -- the ratio of
state traffic to parameter/gradient traffic changes with size, so "2.63x" is a 1M-parameter number and
must not be extrapolated.

### 3.11 NF2/NF3: the packed N-bit format and choosing levels by measurement

**Packing**: n values of `bits` bits are written consecutively from the LSB into `ceil(n*bits/8)`
bytes; value k starts at bit `k*bits`, so **3-bit crosses byte boundaries** (2- and 4-bit do not).
Getting it wrong is silent, so the module ships a self-test that packs a known index sequence and reads
it back.

**Level choice comes from measurement**: on the 32x32 AR model NF beats uniform by 11% at 2 bits
(5.6206 vs 5.6354); at 3 bits NF leads by 0.0003 (inside noise), so 3-bit uses uniform and decoding
stays a multiply-add. Storage is `bits + 32/block`: at block=64 that is 2.5 bits/weight for 2-bit and
3.5 for 3-bit.

**Why not reuse the existing 8-bit index path**: the general path indexes an arbitrary code table with
8-bit indices, which **saves no storage**; saving storage requires narrowing the index width itself --
that is the reason both modules exist.

'''

REPORT_ZH = '''### 3.8 核显（OpenCL/CLBlast）：真实形状 2.6~2.7× CPU，而并发不如顺序

**列主序根因**：CLBlast 在非立方形状上的全部失败（-1011/-1015/-1016）都不是内核问题，而是
**行主序调用方式**违反了它的列主序 API。改成"列主序 + 转置映射"的零拷贝调用后，真实模型形状
全部可用。

**实测（Vega 6 / R5-4500U，fp32）**：MLP up 535.0、MLP down 512.5、QKV 531.5、o proj 371.5、
head 486.0、attn QK/AV 116.7/119.5 GFLOPS；同形状 CPU 164~256。**套件平均 505~551 vs CPU 204
⇒ 2.6~2.7×**，立方 2048³ 655.0 GFLOPS。

**与 GT 1030 的等效性**：fp16 纯运算 1290.1 GFLOPS = GT 1030 理论值（1127）的 **1.14×**；
fp16 GEMM 最高 1039.4（0.92×）；fp32 GEMM 655（0.58×）。**"纯运算达标"不等于"GEMM 达标"**，
两个数字必须分开报。

**设备常驻单层训练步**：2048 宽层 CPU 202.9 ms vs 设备 135.1 ms = **1.50×**（权重相对误差
7.6e-05）；1024 层 1.35×；512 层 CPU 反而快 1.24×。设备端只回传 4 字节 loss。

**何时用核显**：M=1 归 CPU；**M≥64 归核显（约 2×）**。**并发不是答案**：iGPU 保留 40~54%、
CPU 53~61%（共享 L3 被挤），并发 1.1~1.3× < 顺序 1.51×。

**适用性**：实验性、脚本级、未进 wheel；核显训练只在"参数≥16 MB 且 K 大"的层上成立。

### 3.9 4-bit 优化器：内存 1.031 B/参数，速度 1M 2.56×、≥64M 1.74×

| 量 | 实测 |
|---|---|
| 状态内存（1M 参数） | 1.031 B/参数（8-bit 2.062、fp32 8.0） |
| 30.49M 模型 | 232.6 MB → 29.5 MB（7.9×） |
| 内核步（1M 参数） | 0.658 ms vs 8-bit 1.682 ms = 2.56× |
| 内核步（≥64M 参数） | **1.74×**（5 次 1.69~1.81）——早期"2.63×"是 1M 参数值，已收回 |
| 训练轨迹（120 步 AR） | 与 fp32 相差 0.27% |
| 状态码 | 与独立 numpy 50 步逐位一致 |

**结论**：4-bit 优化器是"工具"不是"玩具"——内存确定减半、轨迹同轨、速度在足够大的参数规模上
+74%。但它只在 fp32 参数、blocksize=256、且 DLL 含 4-bit 内核时成立；缺内核时导入即报错。

### 3.10 2-bit/3-bit 量化：损失代价已知，端到端判决是"分两条路"

**损失（32×32 AR 模型，512 条验证样本）**：fp32 5.5038；4-bit +0.004；3-bit +0.019；
2-bit nf +0.117（2.1%）。**2-bit 时 NF 比 uniform 好 11%，3-bit 时 NF 只差 0.0003（噪声）**。

**两条端到端判决**：
- ⚠️ **SD1.5 全 pipeline 用 2-bit 权重：不成立**（存储 −39.5%，图像彻底坏掉）
- ✓ **冻结 2-bit 基座 + LoRA：成立**（只替换 `nn.Linear`，初始验证 loss 5.5538 与格式一致，
  LoRA 可训、loss 下降）

**存储**：2-bit 2.5 bit/权重、3-bit 3.5 bit/权重（block=64），对比 4-bit 的 4.5 与 fp32 的 32。
脚本级组件，未进 wheel。

'''

REPORT_EN = '''### 3.8 iGPU (OpenCL/CLBlast): 2.6-2.7x the CPU on real shapes, while concurrency loses to sequential

**The column-major root cause**: every CLBlast failure on non-cubic shapes (-1011/-1015/-1016) was
**row-major call style** violating its column-major API, not a kernel problem. After switching to the
zero-copy "column-major + transpose mapping" call, every real model shape works.

**Measured (Vega 6 / R5-4500U, fp32)**: MLP up 535.0, MLP down 512.5, QKV 531.5, o proj 371.5,
head 486.0, attn QK/AV 116.7/119.5 GFLOPS; the CPU on the same shapes gets 164-256. **Suite average
505-551 vs CPU 204 => 2.6-2.7x**; cube 2048^3 655.0 GFLOPS.

**GT 1030 equivalence**: fp16 pure compute 1290.1 GFLOPS = **1.14x** the GT 1030's theoretical 1127;
fp16 GEMM peaks at 1039.4 (0.92x); fp32 GEMM at 655 (0.58x). **"pure compute meets the target" does
not mean "GEMM meets the target"** -- the two numbers have to be reported separately.

**Device-resident single-layer step**: a 2048-wide layer measures CPU 202.9 ms vs device 135.1 ms =
**1.50x** (weights agree to 7.6e-05); a 1024 layer 1.35x; a 512 layer has the CPU ahead at 1.24x. The
device returns only a 4-byte loss.

**When to use the iGPU**: M=1 belongs to the CPU; **M>=64 belongs to the iGPU (about 2x)**.
**Concurrency is not the answer**: the iGPU retains 40-54% and the CPU 53-61% (shared L3 thrashed), so
concurrency gives 1.1-1.3x against 1.51x sequential.

**Applicability**: experimental, script-level, not in the wheel; iGPU training pays only on layers with
parameters >=16 MB and a large K.

### 3.9 The 4-bit optimizer: 1.031 B/parameter of state, 2.56x at 1M params and 1.74x at >=64M

| Quantity | Measured |
|---|---|
| State memory (1M params) | 1.031 B/param (8-bit 2.062, fp32 8.0) |
| 30.49M model | 232.6 MB -> 29.5 MB (7.9x) |
| Kernel step (1M params) | 0.658 ms vs 8-bit 1.682 ms = 2.56x |
| Kernel step (>=64M params) | **1.74x** (5 runs 1.69-1.81) -- the early "2.63x" was a 1M-parameter number and is retracted |
| Training trajectory (120 AR steps) | within 0.27% of fp32 |
| State codes | bit-identical to an independent numpy implementation over 50 steps |

**Verdict**: the 4-bit optimizer is a tool, not a toy -- memory is deterministically halved, the
trajectory stays on the same track, and the step is 74% faster at a large enough parameter count. It
only holds for fp32 parameters, blocksize=256 and a DLL that contains the 4-bit kernel; without the
kernel the import fails loudly.

### 3.10 2-bit/3-bit quantisation: the loss cost is known, and the end-to-end verdict splits in two

**Loss (32x32 AR model, 512-sample validation)**: fp32 5.5038; 4-bit +0.004; 3-bit +0.019;
2-bit nf +0.117 (2.1%). **At 2 bits NF beats uniform by 11%; at 3 bits NF differs by 0.0003 (noise).**

**Two end-to-end verdicts**:
- ⚠️ **SD1.5 full pipeline with 2-bit weights: does not work** (storage -39.5%, images destroyed)
- ✓ **Frozen 2-bit base + LoRA: works** (replace only `nn.Linear`; the initial validation loss 5.5538
  matches the format, LoRA trains and the loss drops)

**Storage**: 2-bit 2.5 bits/weight, 3-bit 3.5 bits/weight (block=64), against 4.5 for the 4-bit path and
32 for fp32. Script-level components, not in the wheel.

'''

EDITS = [
    # QUICKSTART
    ('## 5. 硬盘均衡负载（disk_balancer，SSD 保护）', QUICKSTART_ZH + '## 5. 硬盘均衡负载（disk_balancer，SSD 保护）'),
    ('## 5. Disk load balancer (disk_balancer, SSD protection)', QUICKSTART_EN + '## 5. Disk load balancer (disk_balancer, SSD protection)'),
    # TECHNICAL_GUIDE
    ('## 4. 内存策略：为什么「量化冻结层」而不是「降权重存储精度」', GUIDE_ZH + '## 4. 内存策略：为什么「量化冻结层」而不是「降权重存储精度」'),
    ('## 4. Memory strategy: why "quantized frozen layers" not "lower-precision weight storage"', GUIDE_EN + '## 4. Memory strategy: why "quantized frozen layers" not "lower-precision weight storage"'),
    # TECH_REPORT
    ('## 4. 生图训练实测（R5-4500U）', REPORT_ZH + '## 4. 生图训练实测（R5-4500U）'),
    ('## 4. Image-generation training measured (R5-4500U)', REPORT_EN + '## 4. Image-generation training measured (R5-4500U)'),
    # stale base-version claims
    ('## 2. 改动总览（相对 v0.45.1）', '## 2. 改动总览（相对 upstream 0.50.2 / main）'),
    ('## 2. Change Overview (relative to v0.45.1)', '## 2. Change Overview (relative to upstream 0.50.2 / main)'),
]


def main(dry: bool) -> int:
    import os
    files = {
        'QUICKSTART.md': EDITS[0],
        'QUICKSTART_EN.md': EDITS[1],
        'TECHNICAL_GUIDE.md': EDITS[2],
        'TECHNICAL_GUIDE_EN.md': EDITS[3],
        'TECH_REPORT.md': EDITS[4],
        'TECH_REPORT_EN.md': EDITS[5],
    }
    ok = True
    for name, (old, new) in files.items():
        path = os.path.join(DOCS, name)
        text = open(path, encoding='utf-8').read()
        n = text.count(old)
        if n != 1:
            print('ABORT %-24s anchor occurrences=%d' % (name, n))
            ok = False
            continue
        if not dry:
            open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    %-24s +%d chars' % (name, len(new) - len(old)))
    # version headings
    for name, (old, new) in (('TECHNICAL_GUIDE.md', EDITS[6]), ('TECHNICAL_GUIDE_EN.md', EDITS[7])):
        path = os.path.join(DOCS, name)
        text = open(path, encoding='utf-8').read()
        n = text.count(old)
        if n != 1:
            print('ABORT %-24s version anchor occurrences=%d' % (name, n))
            ok = False
            continue
        if not dry:
            open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    %-24s version heading' % name)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main('--dry' in sys.argv))
