"""Append 7.34: the hardware inventory the user asked for, and record the pending tests.

The user's question: what else in the hardware can be squeezed? Answering it properly means
listing all four subsystems and marking each as already-used, already-excluded, or untouched --
not just naming new ideas, because the useful part is knowing which doors are still open.

Also records two things to try when there is time, as the user asked:
  * replacing or bypassing torch's CPU SDPA backend (attention is 18.8% of the step and its
    efficiency has never been measured)
  * measuring the achievable memory-level parallelism, since 277 ns of latency with 27 GB/s
    implies only about 7.5 KB in flight

Nothing here is measured except where it says so. Everything else is marked as an estimate or
as unknown, and the estimates are labelled as estimates.
"""
from __future__ import annotations

import io
import re
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.34 硬件可压榨清单（用户要求记录）

四类资源逐项标注：**已压过 / 已明确否掉 / 还没碰过**。有用的部分不是新点子，而是知道
**哪扇门还开着**。

#### 7.34.1 CPU（R5-4500U · 6 核 6 线程 Zen2 · 满载 2375 MHz）

| 项 | 状态 | 数据 |
|---|---|---|
| AVX2 + FMA 算术 | **已压到 59~81%** | 218（真实形状）~ 371（L2 内）GFLOPS |
| Muon 减少总步数 | **已压，最主要的收益** | **3.57×**（步数 4.98× ÷ 成本 1.37×） |
| NS 迭代 5→3 | **已压** | 优化器省 20.4% |
| 4-bit 优化器状态 | **已压** | 7.8× 内存，内核快 2.56× |
| 超线程 | ✗ **硬件没有**（4500U 是 6 核 6 线程） | — |
| **L3 分块 / 工作集控制** | ★ **部分摸到，未主动利用** | 同一内核：n=384 → 371，n=2048 → 230 |
| flash attention 的 CPU 后端 | ★ **未测** | 占 18.8%，**效率未知** |
| elementwise 融合（490 次 mul） | ★ 未做，**RMSNorm 已融合** | 占 18.8%，剩余目标少 |
| bf16/fp16 前向 | ★ **未知** | **AVX2 无 bf16 指令**，fp16 转换有成本 |

#### 7.34.2 内存（DDR4-2667 双通道 · 标称 42.7 GB/s）

| 项 | 状态 | 数据 |
|---|---|---|
| 频率/时序 | ✗ **BIOS 锁**（用户试过） | 实际 = 标称 = 2667 MT/s，**没有降频** |
| 实测读带宽 | ✗ 27.17 GB/s（6 线程）/ 23.7（4 线程） | 标称的 55~64% |
| **延迟** | ✗ **277.6 ns**（正常 70~90） | 带宽 × 延迟 ⇒ **仅约 7.5 KB 在飞** |
| 线程扩展 | ✗ **3 线程就平了** | 1→16.2 · 2→22.3 · 3→24.8 · 6→27.2 |
| 训练的实际需求 | ✓ **远低于上限**（0.2~0.3 GB/s = 1%） | **所以内存不是训练瓶颈** |

**⇒ 结论：内存对训练【不是】瓶颈，对核显【是】。**

#### 7.34.3 核显（Vega 6 · 6 CU）

| 项 | 状态 | 数据 |
|---|---|---|
| 单独可用性 | ✓ 可跑（非立方 GEMM） | 300 GFLOPS（**基线本身不可复现**，见 7.28） |
| **LDS 吞吐** | ✗ **定罪** | 66~68 GB/s，而需 544 ⇒ **缺口 8 倍** |
| 并发使用 | ✗ **五条全负** | 0.23× / 0.39× / 0.81× / 0.52× / 0.68~0.91× |
| occupancy | ✗ 不是瓶颈 | 组/CU 翻倍 +3%，翻 4 倍 −25% |
| **fixed-function 单元**（视频编解码） | ★ **没碰过，但训练用不上** | 只对视频管线有意义 |
| 寄存器分块（16×8） | ★ **未试** —— **唯一未试的方向** | 可能 300 → 500，但**仍不能并发** |

#### 7.34.4 存储（WD SN730 NVMe 512 GB）

| 项 | 状态 | 说明 |
|---|---|---|
| 训练用途 | ✓ **完全没用上** | token 全在内存（118 MB），每步从 RAM 取 |
| 数据集 | ✓ 1.25 GB / 14432 张 | **整个能进内存** |
| ★ **什么时候会变成瓶颈** | — | **数据集【大到装不下】时**（ImageNet 150 GB 级别） |

**⇒ 存储目前不是资源，因为没有用到它。** 但这也解释了为什么这台机器训不了大数据集。

#### 7.34.5 两个记下来待做的测试（用户要求）

**① torch 的 CPU SDPA 后端有多好？**（约 30 分钟）

```
形状: T=1024 · 8 头 · 4 KV 头 · d=64（我们的真实配置）
对比: F.scaled_dot_product_attention  vs  手写 matmul + softmax + matmul
判据: 若 SDPA 明显更慢 ⇒ 白拿的收益；若已经最优 ⇒ 这条关闭
依据: attention 占一步的 18.8%，而它的效率【从未测过】—— 这是剩下唯一性质未知的大项
上限: 若快 2 倍，整步省约 9.4%
```

**② 内存级并行度到底多少？**（约 30 分钟）

```
已知: 27 GB/s × 277 ns ⇒ 7.5 KB 在飞
对比: 理论 42.7 GB/s 配正常 90 ns 只需 3.8 KB 在飞
要测: 不同【独立流数量】下的带宽 ⇒ 找出在飞请求数的实际上限
判据: 若在飞数远低于硬件能力 ⇒ 还有优化空间；若已到顶 ⇒ 内存墙确认
注意: 先【把 277 ns 测准】—— 它比正常值高 3 倍，可疑，且一切推断都建立在它上面
```

#### 7.34.6 总判断

**CPU 侧最大的那块已经拿走了**（Muon，3.57×）。剩下的候选都在**个位数百分比**量级：
attention 后端最多约 9.4%，elementwise 最多约 3.8%，两者相加的上限约 13%，即 3.57× → 约 4.0×，
**且需要两件事都做到最好**。

**内存与核显的结论都是负的，且原因已定位到物理层**（BIOS 锁的分频/延迟；UMA 无独立显存）。
**负结论也是结论**：它省掉了未来在这些方向上的反复尝试。

**要再有量级的突破，需要的不是优化而是换硬件**：LPDDR5 平台（68~100 GB/s）或一张带独立
显存的便宜独显。这一点在 7.28 已经写过，此处只是再次确认它没有被后来的发现推翻。

'''

m = re.search(r'^### 7\.34 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.34'
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
