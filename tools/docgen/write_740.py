"""Append 7.40: the GPU technique list mapped onto our measured bottlenecks.

The user found a list of GPU techniques and asked which apply. The useful part is not the list
but the correspondence: four of the ten land directly on bottlenecks we have already measured,
and three are either inapplicable or actively describe our problem rather than a solution.

The deepest point, which is worth recording as a direction rather than an experiment: the whole
GPU design philosophy is explicit programmer-managed on-chip data. Every optimisation this
project has done so far -- Muon, weight decay, quantisation -- is orthogonal to that. And all
three of our open bottlenecks (371 vs 230 GFLOPS decided by L2 residency, 66 against 544 GB/s of
LDS, 277 ns of latency with 7.5 KB in flight) have the same root: data is not where it needs to
be when it is needed.
"""
from __future__ import annotations

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.40 GPU 技术清单与我们瓶颈的对应（用户搜集）

用户在检索时找到十项 GPU 技术。**真正有用的不是清单本身，而是对应关系**：其中四项直接落在
我们已测出的瓶颈上，三项要么不适用、要么**描述的正是我们的病灶而不是药方**。

#### 7.40.1 可用的四项

| 技术 | 打我们的哪里 | 依据 | 成本 |
|---|---|---|---|
| **Swizzle 地址重排** | **核显 LDS** | **66 GB/s 而需 544**。此前七次失败改的全是**访问方式**（转置 / float4 / 列交错），**没有一次改地址映射**——这是不同的事 | 半天 |
| **Scratchpad · On-Chip SRAM · Render Pass** | **CPU 的 L2 落差** + attention | 三者是**同一个原理**：显式管理片上数据。我们的数字是它的反面教材 | 一天 |
| **Async Copy 异步拷贝** | **内存延迟** | **277 ns，在飞仅约 7.5 KB，3 线程饱和**。CPU 无 `cp.async`，等价物是**软件预取** | 半天 |
| **Software-Managed Data Flow** | 概念层 | 编译期显式调度数据流；CPU 上等价于**控制算子顺序与分块粒度** | 与上条一体 |

**"Scratchpad / On-Chip SRAM / Render Pass"这一组值得单独强调**，因为我们的测量正好证明了
这个原理：

```
同一份内核，只改矩阵大小:
  n=384（工作集在 L2 内）⇒ 371 GFLOPS
  n=2048（溢出到内存）  ⇒ 230 GFLOPS
```

**1.6 倍的落差，完全由"工作集在不在片上"决定。** 而**我们从未试过为 L2 驻留做分块**。

#### 7.40.2 不适用、且其中一条是病根

| 技术 | 判断 |
|---|---|
| **Unified Memory 统一内存** | ★ **这正是我们核显失败的【原因】** —— APU 就是统一内存架构，核显与 CPU 抢同一条 DDR4，**五条并发实验全负（0.23×~0.91×）**。在独显上是优点，**在我们这里是病根，要从"可抄"里划掉** |
| Dynamic Caching | NVIDIA Ada 的动态片上分配，CPU 无对应物 |
| Tiled Resources | 稀疏虚拟纹理；除非做块稀疏权重，否则用不上 |

#### 7.40.3 精炼后的排序

| 优先级 | 抄什么 | 打哪里 | 成本 |
|---|---|---|---|
| **①** | **Swizzle / padding** | **核显 LDS（66 对 544）**——七次失败里唯一没试过的维度 | 半天 |
| **②** | **为 L2 驻留分块** | **CPU 的 371 对 230** | 一天 |
| **③** | **attention 分块**（FlashAttention） | **占 18.8% 而从未优化** | 半天 |
| **④** | **软件预取** | **277 ns 延迟、7.5 KB 在飞** | 半天 |
| **⑤** | Kernel Fusion（torch.compile） | 490 次 `aten::mul` | 一行 |

#### 7.40.4 而最深的一条：三条瓶颈是同一个根源

> **GPU 的全部设计哲学是"让程序员显式管理片上数据"。**

而本项目到目前为止做的所有优化——**Muon（3.57×）· 权重衰减 · 4-bit 状态 · 量化**——
**没有一条碰过这件事。**

而三个未解决的瓶颈：

| 瓶颈 | 数字 | 本质 |
|---|---|---|
| CPU FMA | **371 → 230 GFLOPS**，只由工作集位置决定 | 数据没在片上 |
| 核显 LDS | **66 对 544 GB/s** | 数据没在片上 |
| 内存延迟 | **277 ns，在飞 7.5 KB** | 数据没在路上 |

**⇒ 三条，同一个根源：数据没有被及时放到该在的地方。**

**这比"再抄一个具体技巧"更重要**——它说明前面那些优化（优化器、正则化、精度）虽然有效，
**但都在另一条轴上**。而这一条轴，一行代码都还没动过。

#### 7.40.5 并入 §7.39 的队列

**§7.39.3 的执行顺序不变，本节的五项插在它的第 ⑤ 步之前**：

```
① 7.36 四项重测（地基，机器必须完全空闲）
② 核显真实上限（300 GFLOPS 不可复现）
③ LDS 微基准独立复现
④ memory-level parallelism
⑤ ★ 本节新增：Swizzle → L2 分块 → attention 分块 → 软件预取
⑥ 然后才是 torch.compile / QKV 融合 / NS 隔步
```

**判据仍按 7.39.4 的纪律：每个"抄"的动作配事先写好的判据。**

'''

m = re.search(r'^### 7\.40 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.40'
else:
    a = re.search(r'^# 附录', s, re.M)
    if a:
        s = s[:a.start()] + SEC + '\n\n' + s[a.start():]
        how = '插在附录之前'
    else:
        s = s.rstrip() + '\n\n' + SEC
        how = '追加'
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('  %s；报告 %d 行' % (how, s.count('\n')))
