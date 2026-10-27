"""Add the new components to docs_cpu/TECHNICAL_GUIDE.md.

Three things are missing from the technical guide and are new since it was written: the 4-bit optimizer,
the 2/3-bit quantisation path, and the iGPU conclusions that replaced the earlier "DirectML is a
per-operator regression" position. The first two are additions; the third is a correction, and is
written as one rather than as a quiet revision.

Inserted before section 4, after the existing 3.8 on the DirectML executor, so the numbering stays and
nothing already written is moved.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

P = r"D:\work\bitsandbytes-CPU\docs_cpu\TECHNICAL_GUIDE.md"
s = io.open(P, encoding="utf-8").read()

ANCHOR = "## 4. 内存策略"
if ANCHOR not in s:
    raise SystemExit("锚点没找到：%s" % ANCHOR)

NEW = '''### 3.9 4-bit 块优化器（AdamW4bit，本 fork 新增）

**状态：已实现、已验证、已发布（`0.50.2.dev3`）。**

**入口：`from bitsandbytes.optim.adamw4bit import AdamW4bit`（或 `bitsandbytes.optim` 已导出时直接导入）。**

#### 3.9.1 它买什么

| 对手 | 内存 | 速度 |
|---|---|---|
| **对 fp32 AdamW** | **7.8×** 少（1.031 对 8.000 字节/参数） | 0.96× |
| **对 AdamW8bit** | ★ **2.06×** 少（1.031 对 2.125） | ★ **2.63× 更快** |

**★ 注意基线。通常引用的 7.8× 是【对 fp32】；而 bnb 用户手里已经是 8-bit，
对他的实际收益是 2.06×。两个都对，回答的是不同的问题。**

#### 3.9.2 而它是【两边都赢】，不是"用速度换内存"

**早期版本的文档写着"没有 AVX2 路径、约慢 4×"，而那两句都是错的。实测：**

```
25 轮交替 × 10 次 · 1M 参数:
  8-bit 中位 1.485 ms · 4-bit 中位 0.513 ms
  配对比较比值中位 2.627 · 标准误 0.253 · 95% CI 2.131 ~ 3.123
⇒ ★ 4-bit 快 2.63×（CI 排除 1）
```

**⇒ 而 AVX2 路径是【存在】的：`optimizer_4bit_blockwise_avx2` 在 `cpu_ops.cpp:3056`，
而分派器 `cpu_ops.cpp:3707` 对【两状态家族】（adam / ademamix）选它 —— 而 `AdamW4bit` 用的正是 adam。
一状态家族（sgd/lion）走标量，那是【刻意的设计】以缩小新代码面。**

#### 3.9.3 正确性

**内核与 numpy 参考【逐码一致】（`test_optimizer_4bit_cpu.py`）。**

**⇒ 而正确性有一个前提不能省：状态按【成对处理】。
逐元素 get/set 会对同一个字节做"读-改-写"，产生 store-to-load forwarding 停顿，
实测比 8-bit 版【慢 4.6×】（7.385 ms 对 1.591 ms / 1M 参数）。
⇒ ★ 所以"一次读字节、拆两个码、算完打包写一次"是性能关键，不是风格。**

#### 3.9.4 在什么规模上用

**⇒ 用户须知：它的收益随参数量线性放大。**

```
30.5M 参数  ⇒ 优化器状态省 203 MB = 内存的 1.3%  ⇒ ★【收益很小】
300M 参数   ⇒ 约省 2 GB = 约 13%                ⇒ ★【才是它的战场】
```

**⇒ 在 30.5M 这个规模上，"换成更大 batch"的收益本来就小。它是为【大模型】准备的，
而不是一个"到处都该换"的默认。**

#### 3.9.5 用法

```python
from bitsandbytes.optim.adamw4bit import AdamW4bit
opt = AdamW4bit(model.parameters(), lr=1e-4)
```

**⇒ 与 `torch.optim.AdamW` 同签名。实测在 30.5M 模型上训 300 步，
loss 轨迹与 `AdamW8bit` 差 0.53%（最差一格）/ −0.10%（末步）。**

---

### 3.10 2-bit / 3-bit 量化（`quant2bit.py` / `quantnbit.py`）

#### 3.10.1 它们在哪

| 文件 | 覆盖 |
|---|---|
| `quant2bit.py` | 2-bit 专用路径 |
| `quantnbit.py` | 通用 N-bit（2/3/4/5 位），带自检往返 |
| `quant_lora.py` | 接入训练（`nf2` / `uni3` / `nf3` 等分支），冻结基座 + LoRA |
| `linear2bit.py` · `embed2bit.py` | 层替换入口 |

**⇒ 每字节 4 个索引，位序：字节内第 k 个值占 `[2k, 2k+1]` 位。**

#### 3.10.2 ★ 而一条重要的负面结论（已有记录，此处重申）

**"把基座权重降精度【存储】"这条路的实测结论是【负面的】——
详见 `TECH_REPORT.md` §3.5。而它与 4-bit 【优化器】状态的结论【不冲突】，因为两者改的不是同一个东西：**

```
✗ 降【权重】精度  ⇒ 影响【每次前向】的数值与解量化开销 ⇒ 实测负面
✓ 降【优化器状态】精度 ⇒ 只影响【更新步】的量化误差 ⇒ 而它被证明是快的
⇒ ★【所以"4-bit 好"这个说法必须带主语:4-bit 的是【优化器状态】,不是权重】
```

---

### 3.11 ★★ 核显：从"逐算子负优化"到"设备常驻 + 有明确适用区间"

**§3.8 记录的是 DirectML 块级常驻执行器（实验性，默认关闭）。
此后对这颗核显（Vega 6 / gfx902，6 CU，384 shader）的测量推翻了一条、建立了一条，两者都写在这里。**

#### 3.11.1 ★ 被推翻的：不是功耗仲裁，是共享 L3

**早期结论把"CPU 一忙核显就掉 5×"归因为【15W 封装里的功耗仲裁】（一个开关式约束）。
而那个归因是错的，而错在读错了曲线形状。**

实测（线程数固定为 1，只改 CPU 的工作集）：

| CPU 负载（1 线程） | 工作集 | 核显相对空闲 |
|---|---|---|
| `torch.mm` 256³ | 0.8 MB | **1.96×** |
| `torch.mm` 1024³ | 12 MB | **3.76×** |
| `torch.mm` 2048³ | 48 MB | ★ **6.08×** |
| 纯 ALU 自旋（无访存） | 0 | ★ **1.03×** |

**⇒ ★【满功耗的纯计算只掉 3%，而工作集越大掉得越多
⇒ 机制是【共享的 8 MB L3 被 CPU 的工作集挤掉】，不是功耗。】**

**⇒ 而它的意义完全不同：一个开关式的功耗约束只能靠"加功率"绕开（那对本机不可接受），
而一个连续的缓存效应是【可调的量】。**

#### 3.11.2 ★ 而建立的：不是分工，是换手

**扫 M（N=6144, K=2048，Qwen3 gate_proj 的真实形状）：**

| M | 核显单独 | CPU 单独 |
|---|---|---|
| 1 | 5.3 | ★ **11.7** |
| 64 | ★ **356.8** | 56.7 |
| 512 | ★ **409.2** | 159.8 |
| 2048 | ★ **425.3** | 199.7 |

**⇒ ★【M=1 给 CPU（单 token 无法并行），M≥64 给核显（约 2 倍）。
而"两者一起"只比核显单独高 7% —— 因为共享 L3 是【零和的】。】**

**⇒ ★ 所以正确的形态是【换手】，不是【分工】**：
**小 M 用 CPU，大 M 用核显，而【不要让它们同时干同一份活】。**

#### 3.11.3 ★★★ 而设备常驻训练：技术上成立，适用区间明确

**已实现并验证：**
* OpenCL GEMM 作为 `torch.autograd.Function`，前向与反传都在核显
* **梯度等价：`dX` 1.3e-06 · `dW` 1.0e-06（对 torch 的 autograd）**
* Adam 内核：与 torch 逐步对拍 5 步，误差 1.7e-07；**转置副本【逐位精确】**
* **端到端训练与 CPU 【逐位一致】（loss 轨迹与权重变化量完全相同）**
* 设备常驻后，每步主机流量从 **O(NK)** 降到 **O(BK+BN)**

**→ 而它的适用区间是被测过的，不是估计的：**

| N × K × B | 参数 | CPU ms | 设备 ms | 判定 |
|---|---|---|---|---|
| 512×512×1024 | 1.0 MB | 7.24 | 13.28 | ★ **CPU 赢 1.84×** |
| 1024×1024×1024 | 4.2 MB | 40.68 | 37.69 | 平 |
| **2048×2048×1024** | 16.8 MB | 152.56 | **117.42** | ★ **设备赢 1.30×** |
| **2048×2048×4096** | 16.8 MB | 456.01 | **347.71** | ★ **设备赢 1.31×** |
| 4096×1024×4096 | 16.8 MB | 570.55 | 594.87 | 平 |

**⇒ ★【而三条读数各有含义：**
* **参数越大越有利**（省的是 O(NK)）
* **batch 几乎无影响**（1024 → 4096 时 1.30 → 1.31）
* ★ **而 4096×1024 平 ⇒ 还要 K 也大**（核显需要长的归约维）
**⇒ 所以适用区间是：【参数 ≥ 16 MB 且 K 大】。**

#### 3.11.4 ★ 一条使用提醒

**这颗核显的结论【必须以"形状 × 内核 × 状态"三维读】，不能压成一个数。
本项目在同一条线上量错过四次，而两次的方向还相反：**

```
入队时间当执行时间   ⇒ 高估 40 倍（46787 GFLOPS，而设备峰值 1150）
每步重传权重当常态   ⇒ 低估 5.8 倍（64.5 对 374 GFLOPS）
```

**⇒ ★【判据：任何超过设备标称峰值的核显数字，先当测量错误处理，而不是当发现。】**

'''

s = s.replace(ANCHOR, NEW + ANCHOR, 1)
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("  ✓ 已插入 3.9 / 3.10 / 3.11 三节")
print("  ✓ TECHNICAL_GUIDE.md 现在 %d 行" % s.count("\n"))
