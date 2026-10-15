"""Append 7.45: swizzle and register blocking, and the finding that they are substitutes.

Two optimizations, each measured alone and then together, and the combination is worse than
either one alone. That is the interesting part and it has a clean explanation.

    configuration                       reads/FMA   GFLOPS   ratio
    4x4, no swizzle (the fixed v4)      0.500       183.4    1.00
    4x4 + As17 + swizzle                0.500       349.9    1.91
    8x8, no swizzle                     0.250       271.1    1.48
    8x8 + As17 + swizzle                0.250       320.3    1.75   <- worse than swizzle alone
    4x4 + As17 + swizzle + B65          0.500       361.5    1.97

Register blocking was only ever compensating for the bank conflicts. Once the conflicts are gone,
LDS delivers enough per access that halving the *number* of accesses buys nothing, while the cost
of 64 accumulators per thread -- lower occupancy -- is still paid. So the two are substitutes.

This also overturns the conclusion recorded in 7.43 this morning, that the iGPU is slower than the
CPU. That rested on 170 GFLOPS, which was the kernel *before* the addressing fix. At ~360 GFLOPS it
is about 1.5x the CPU's 218-240.

Which reopens something else: all five concurrency experiments were run with the slow kernel. The
NS offload result of 0.23x in particular assumed a device that could not beat the CPU at all.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.45 Swizzle 与寄存器分块：两个优化，而它们【互相替代】不是叠加

这一节推翻了三处此前的结论，包括今天上午刚写的。

#### 7.45.1 分别测量

三个变体，形状 1024×2736×512（真实形状），正确性每次对 numpy 校验：

| 配置 | 读/FMA | GFLOPS | 相对 |
|---|---|---|---|
| 4×4 基线（修好越界后的 v4） | 0.500 | **183.4** | 1.00× |
| **4×4 + As 行距 17 + Bs swizzle** | 0.500 | **349.9** | **1.91×** |
| **8×8 仅寄存器分块** | 0.250 | **271.1** | **1.48×** |

**两个方向各自有效，而 swizzle 效果更大。**

#### 7.45.2 而合起来【更慢】

| 配置 | 读/FMA | GFLOPS | 相对 |
|---|---|---|---|
| **8×8 + As17 + swizzle** | 0.250 | **320.3** | 1.75× ✗ |
| 8×8 + As17 + swizzle + B65 | 0.250 | 319.7 | 1.74× ✗ |
| **4×4 + As17 + swizzle + B65** | 0.500 | **361.5** | **1.97×** ✓✓ 最佳 |

**⇒ 合起来 320 反而【低于】只做 swizzle 的 350。**

#### 7.45.3 为什么：它们是替代品

**寄存器分块【本来就是在补偿 bank conflict】**：既然每次访问都因冲突而变慢，那减少访问次数
就能找回一部分。**而冲突一旦消掉，LDS 每次访问就能给足带宽，减少访问次数便不再有收益**，
而 64 个累加器带来的占用率下降**仍然要付**。

**⇒ 所以正确的做法是消冲突（swizzle），不是减访问量（分块）。** 这也解释了为什么我此前
七次内核尝试全部失败——**它们改的都是访问方式或访问量，没有一次动地址映射。**

#### 7.45.4 核显的真实数字：约 360 GFLOPS，CPU 的 1.5 倍

| 项 | 数值 |
|---|---|
| **核显（swizzle，最佳配置）** | **361.5 GFLOPS** |
| CPU 同形状 | 218~240 GFLOPS |
| **比值** | **核显快约 1.5 倍** |
| 修好越界但未 swizzle | 170~185 GFLOPS |

**⇒ 7.43 今早写的"核显比 CPU 慢（0.86×）"【作废】** —— 那个数字取自**地址映射修复之前**的
内核。核显实际更快。

**判据（7.39.4 事先固定）：≥ 400 GFLOPS 才判定"LDS 不是硬墙"。实测 361.5，【未达 400】**
⇒ 因此严格按判据，**不能说 LDS 完全不是问题**。但 361.5 已远超 CPU，**所以 7.28.6 那个
"缺口 8 倍"的判断肯定错了**——真实缺口最多 2 倍，而且可以用 swizzle 消掉。

#### 7.45.5 而这重新打开了并发那件事

**五条并发实验全部用的是【未修复的慢内核】**：

| 实验 | 当时结果 | 前提 |
|---|---|---|
| NS 同步交替 | 0.23× | 核显 170 GFLOPS |
| 全设备化 | 0.39× | 同上 |
| **真异步** | 比串行差 5% | 同上 |
| 两个独立任务 | 0.81× | 同上 |
| 切分同一矩阵 | 0.68~0.91× | 同上 |

**⇒ 而现在核显的 GEMM 是 361.5，而 CPU 是 240 ⇒ 核显【单独就能快 1.5 倍】。**
**⇒ "卸载给它"这笔账的前提完全变了**：以前是"把一个 CPU 更快的工作交给一个更慢的设备"，
现在是"交给一个更快的设备，代价是抢内存"。

**所以下一步应当是：用 swizzle 后的内核重测并发。** 判据同样是事先写好的：
**若并发后的总吞吐 ≥ CPU 单独 1.2 倍 ⇒ 核显这条路【重新打开】；否则仍然关闭。**

#### 7.45.6 而正确的表述是

**7.28 的六条证据里，现在有两条被改动**：

* ~~单内核速度 300 GFLOPS~~ ⇒ 170（未 swizzle）⇒ **361.5（swizzle 后）**
* **LDS 缺口 8 倍** ⇒ 实测**最多 2 倍，且可被 swizzle 消除**

**其余四条不变**（五条并发用的是慢内核，现在要重测；occupancy；小矩阵；UMA 架构）。

**⇒ 而 UMA 那条仍然成立，它才是"能不能并用"的根本约束。** 361.5 只说明核显单个内核更快，
不说明两个能一起用——**而那正是要重测的。**

'''

m = re.search(r'^### 7\.45 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.45'
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
