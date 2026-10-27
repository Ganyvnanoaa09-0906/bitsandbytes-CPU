"""Add the 4-bit optimizer to QUICKSTART's feature list, and correct the iGPU entry.

QUICKSTART section 4 lists the practical features one per subsection, 4.1 through 4.8, and the 4-bit
optimizer is not among them. A component that works but is not in the quick start is a component nobody
finds -- which is the same failure as the header comment that said the kernel had no AVX2 path: the
thing is fine and the description is what stops it being used.

4.9 is added for the optimizer, and 4.8's iGPU entry is corrected, because it describes the executor as
the recommended form and the measurements since have changed what the recommendation should be.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

P = r"D:\work\bitsandbytes-CPU\docs_cpu\QUICKSTART.md"
s = io.open(P, encoding="utf-8").read()

# ---------- 1) 在 §5 之前插入 §4.9 ----------
ANCHOR = "## 5. 硬盘均衡负载"
if ANCHOR not in s:
    raise SystemExit("锚点没找到: %s" % ANCHOR)

NEW49 = '''### 4.9 ★ 4-bit 块优化器（AdamW4bit，本 fork 新增）

**降低【优化器状态】的内存，而不是降低权重精度 —— 后者已被实测否定（见 `TECH_REPORT.md` §3.5）。**

#### 为什么用它

| 对手 | 内存 | 速度 |
|---|---|---|
| 对 fp32 AdamW | **7.8×** 少（1.031 对 8.000 字节/参数） | 0.96× |
| **对 AdamW8bit** | ★ **2.06×** 少（1.031 对 2.125） | ★ **2.63× 更快** |

**★ 两个基线都要说：7.8× 是对 fp32 的，而你手里已经是 8-bit，所以实际收益是 2.06×。
两个数都对，回答的是不同的问题。**

**★ 而它是【两边都赢】，不是"用速度换内存"。**

#### 用法

```python
from bitsandbytes.optim.adamw4bit import AdamW4bit
opt = AdamW4bit(model.parameters(), lr=1e-4)
```

**⇒ 与 `torch.optim.AdamW` 同签名。**

#### ★ 在什么规模上用 —— 这条比上面所有数字都重要

```
30.5M 参数  ⇒ 优化器状态省 203 MB = 内存的 1.3%   ⇒ ★【收益很小】
100M 参数   ⇒ 约省 400 MB         ≈ 内存的 2.5%   ⇒ 仍然小
300M 参数   ⇒ 约省 2 GB            = 内存的 13%    ⇒ ★【才是它的战场】
```

**⇒ 在 30.5M 这种规模上，"换成更大 batch"的收益本来就很小。
它是为【大模型】准备的，不是一个"到处都该换"的默认。**

#### 实测（30.5M 模型 · 真实数据 · 300 步）

```
loss 轨迹   与 AdamW8bit 差 −0.10%（末步）· +0.53%（最差一格）
步时        0.955×（4-bit 反而略快）
状态内存    66.3 MB → 33.4 MB ⇒ 2.0×
```

**⇒ 正确性：内核与 numpy 参考【逐码一致】（`test_optimizer_4bit_cpu.py`）。**

#### 注意事项

* **AVX2 路径存在**（`optimizer_4bit_blockwise_avx2`，`cpu_ops.cpp:3056`），
  分派对【两状态家族】（adam/ademamix）选它；一状态家族（sgd/lion）走标量，那是刻意设计。
* **不要逐元素 get/set 状态**：那会产生 store-to-load 停顿，实测慢 4.6×。
  内核对同一个字节"读一次、拆两个码、写一次"，这是性能关键而非风格。

---

'''

s = s.replace(ANCHOR, NEW49 + ANCHOR, 1)

# ---------- 2) 修正 §4.8 的核显条目 ----------
OLD48_HEAD = "### 4.8 核显（DirectML）块级常驻执行器（实验性，默认关闭）"
NEW48_HEAD = '''### 4.8 核显（DirectML）块级常驻执行器（实验性，默认关闭）

> **⚠️ 2026-10-17 补充：该组件的定位已变化，选用前先读 `TECHNICAL_GUIDE.md` §3.11。**
>
> 此后对这颗核显（Vega 6 / gfx902）的测量给出了三条新的结论：
>
> * **"CPU 一忙核显就掉 5×" 的机制是【共享 8 MB L3 被挤】，不是功耗仲裁**
>   （纯 ALU 满转只掉 3%；而 48 MB 工作集掉 6×）
> * **正确的用法是【换手】不是【分工】**：M=1 给 CPU，M≥64 给核显，两者一起只多 7%
> * ★ **设备常驻训练已实现并与 CPU【逐位一致】**，而它的适用区间是
>   **参数 ≥ 16 MB 且 K 大** ⇒ 在那个区间赢 **1.30×**；512×512 那种小层上 CPU 仍赢 1.84×
>
> **⇒ 所以在小层/小 batch 上，这个执行器【不该开】。判据是形状，不是"有没有核显"。**'''

if OLD48_HEAD in s:
    s = s.replace(OLD48_HEAD, NEW48_HEAD, 1)
    print("  ✓ §4.8 已加更正说明")
else:
    print("  ? §4.8 锚点没找到")

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("  ✓ §4.9 已插入（4-bit 优化器）")
print("  ✓ QUICKSTART.md 现在 %d 行" % s.count("\n"))
