"""Append 7.51: the mathematical route to the 10x, and why verification comes first.

The framing the user arrived at: the per-instruction question is settled. On Zen2 an FMA already
has the same throughput as an ADD, two per cycle each, and the remaining 19 percent is issue bubbles
that SMT would fill -- except the 4500U has no SMT, being a 6-core 6-thread part. So the instruction
level is a dead end and the only lever left is mathematics: make the same function require fewer
scalar operations.

Four families, each a genuine change of asymptotic complexity rather than a constant factor. And
they compose, which matters because the epoch arithmetic says two-fold savings are useless -- 0.43
epochs doubled is still under one.

The section also fixes a rule for how such a replacement is to be evaluated, and the first item is
not a benchmark. A factored matrix multiplication that computes something other than the original
is faster for the trivial reason that it is doing less work, and every failure this project has had
came from measuring something that looked right.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.51 数学路线：把"10 倍"从复杂度里要出来

#### 7.51.1 前提：指令层面确实到头了

```
Zen2 每个核 4 条 FP 流水线: FP0/FP1 做 256 位 FMA · FP2/FP3 做 256 位 ADD
VADDPS 吞吐 【2/周期】· VFMA 吞吐 【2/周期】
⇒ ★【FMA 在吞吐上已经和加法一样便宜】
差别只剩延迟（4 对 3）与端口,而延迟被 12 个独立累加器盖住了
```

**而剩下的 19%（371 对 456 峰值）是发射气泡，正常靠 SMT 填 —— 而 4500U 是【6 核 6 线程，没有 SMT】**
（Renoir 的 4500U/4700U 均为无超线程设计）。**⇒ 结构性死角，指令层面无路。**

**⇒ 所以只能从数学要：让同一个函数需要更少的标量运算。**

#### 7.51.2 四族，而它们能【相乘】

| 族 | 复杂度变化 | 我们的尺寸上的增益 |
|---|---|---|
| **低秩 W≈UV** | mn → r(m+n) | MLP 512→2048、r=128：2.10M → 0.66M ⇒ **3.2×** |
| **★ Butterfly / Monarch** | **n² → n·log₂n** | **n=512：262144 → 4608 ⇒ 57×** |
| **Linformer attention** | T² → T·k | k=256：67M → 16.8M ⇒ **4×**（打 attention 的 18.3%） |
| **卷积走 FFT**（VQGAN） | n²k² → n²log n | 大，但 VQGAN 不是训练瓶颈 ⇒ 优先级最低 |

**⇒ 而关键在于它们是可乘的**，而**记忆测试那条算出来的坎是"手段必须值 10 倍以上"**
（0.43 个 epoch，2 倍 ⇒ 0.86，还是不到一个）。

**⇒ 所以这不是"再啃 2%"，而是唯一能真正跨过那道坎的路线。**

#### 7.51.3 Monarch 是什么，为什么它正对着我们的问题

**数学事实**：一个 n×n 稠密矩阵可分解为 **log₂n 个稀疏（2 点）因子的乘积**。

```
dense n×n   ⇒ n²
butterfly   ⇒ n·log₂n
Monarch     ⇒ M = P·L·P·R（两个块对角）⇒ n×n 的 O(n^1.5)
```

**★ 而这一类方法被发明出来的唯一目的就是"让矩阵乘在硬件上便宜"** —— 它是**数学结构**，
不是指令技巧，正好对应用户的判断。

**★★ 而它有一个额外的好处**：因子是**结构化稀疏**的，所以它**同时降低运算量和内存流量**。
**⇒ 它顺手打了今天最根本的那个瓶颈**（371 对 230，只由工作集在不在片上决定）。

#### 7.51.4 而评测这类替换有一条必须放在最前面的规则

**★ 第一步不是性能测试，是【数学等价性验证】。**

**理由**：一个"算的不是原来那个东西"的分解矩阵乘，**会更快，而快的唯一原因是它干得更少。**
**而这正是本项目今天犯了九次的同一种错** —— 测量本身没错，但量的不是要量的那个东西。

**⇒ 所以每个替换都必须先证明：在给定参数下，分解形式与前向【逐位可复现】**
（或至少误差 < 1e-5，且误差随 rank 增大而【单调下降】——**这一条能抓出"写错了但看起来在跑"**）。

#### 7.51.5 实验设计

**替换对象**：MLP 的两层全连接（`w12` 宽 2736 与 `w3`），它们是 GEMM 的主体。

| 步骤 | 内容 | 判据 |
|---|---|---|
| **①** | **等价性**：Monarch 形式在 rank = 全秩时与前向一致 | 误差 < 1e-5 且**随 rank 单调下降** |
| **②** | **速度**：测 FLOPs 与实测步时 | 步时降幅应接近 FLOPs 降幅（若只降一半，说明内存流量成了新瓶颈） |
| **③** | **质量**：rank 扫描（512 / 256 / 128 / 64），各训 2000 步 | 记录 (FLOPs 降幅, val 升幅) 的**交换比** |
| **④** | **决策** | 若某 rank 下 **MLP 快 ≥ 5× 而 val 只涨 ≤ 0.05** ⇒ 换；若 val 涨 ≥ 0.5 ⇒ 换回来 |

**⇒ ③ 的判据必须事先固定**，否则会在"再降一点看看"里失去基准 —— 而那正是 7.34 记录过的问题。

#### 7.51.6 与记忆测试结论的关系

**这一步是【条件分支】，不是并列项：**

| 记忆测试结果 | 意味着 | Monarch 实验 |
|---|---|---|
| **100 张能记住** | 架构能拟合，**只是算力不够** | ★ **正解** —— 数学降算力需求是唯一同时满足质量和时间的解 |
| **100 张仍只有纹理** | 架构或采样有结构性问题 | **先修架构**；此时削表达力只会更糟，**不应做** |

**⇒ 所以顺序是：记忆测试 ⇒ 若为正 ⇒ Monarch。** 而已排的队列（任务分流 / L2 分块 / FMA 四因）
都属于"确认边界在哪"，**它们不改变这个分支，只是把边界描述得更准。**

#### 7.51.7 一句总结

> **硬件给不了的，数学可以给 —— 而数学给的方式是把 n² 变成 n·log n。**

**⇒ 而这条路的意义不只是快：它让"训够一个 epoch"从 17 小时变成可能进到小时级，
而那是这台机器上唯一能真正回答"这套架构行不行"的方式。**

'''

m = re.search(r'^### 7\.51 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.51'
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
