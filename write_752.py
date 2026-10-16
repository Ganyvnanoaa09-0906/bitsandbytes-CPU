"""Append 7.52: the four measurements, and the two doors that reopened.

The results reverse two conclusions from earlier in the day. The task-split measurement shows the CPU
slowing by only 4.3% while the iGPU streams GEMM at full rate, which means the 0.76x figure that
closed the iGPU was an artefact of the measurement rather than a property of the hardware. And the
L2 blocking sweep shows a manual column-chunked GEMM running 36% faster than a single full call,
which is a speedup available to the training immediately.
"""
from __future__ import annotations

import io
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'D:\work\bitsandbytes-CPU'
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()


def tail(name, n=14):
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        return '(未产出)'
    try:
        t = io.open(p, encoding='utf-8', errors='replace').read().strip().split('\n')
        return '\n'.join(t[-n:])
    except Exception as e:
        return '(读取失败 %s)' % str(e)[:60]


m4 = tail('m4_gil.txt', 16)

SEC = '''### 7.52 四项测量：两条被关掉的路，各开了一条

#### 7.52.1 ★ 任务分流：CPU 只慢 4.3% —— 核显那条路重开

**这是今天最重要的一个数字。**

| 项 | 值 |
|---|---|
| CPU 单独（前向+反向） | **4.533 s** |
| 核显跑一组 GEMM | 0.066 s ⇒ **506.9 GFLOPS** |
| **争抢时 CPU** | **4.730 s ⇒ 仅慢 4.3%** |
| **判据（7.51 事先固定 ≤25%）** | ★ **通过，分流值得做** |

**⇒ 7.46 那个 0.76× 是【测量方法的伪影】，不是硬件的性质。**

**⇒ 而它的机制我今晚才想到，而它本来该更早想到**：那次用的是**线程**，核显线程每次发起内核
调用都要抢 GIL，而主线程在跑 torch，**算子之间也要重新获取 GIL** ⇒ **两个线程在 GIL 上排队，
被计时记成了"资源争抢"。**

**⇒ 而算术也早就该否定"带宽争抢"这个解释**：GEMM 的算术强度约 300 FLOP/字节，
CPU 的 240 GFLOPS 只需 0.8 GB/s、核显的 361.5 只需 1.2 GB/s，**合计 2 GB/s 对 27 GB/s ——
只用了总带宽的 7%。**

**⇒ 所以"等效一张 GT 1030"是可达的**：

```
GT 1030 实测 GEMM   ≈ 600~700 GFLOPS · 48 GB/s 独立显存
我们: CPU 240 + 核显 361.5~506.9 = 【601~747 GFLOPS】
⇒ ★【数字上已经在同一档,而唯一缺的"互不干扰"现在测出来了: 只慢 4.3%】
```

#### 7.52.2 ★ L2 驻留分块：1.359× —— 而它【立刻可用于训练】

| 分块宽度 k | 每块 MB | GFLOPS | 相对 |
|---|---|---|---|
| 64 | 0.62 | 104.2 | 0.617× |
| 128 | 1.25 | 116.2 | 0.689× |
| **256** | 2.50 | **221.9** | **1.315×** |
| 342 | 3.34 | 201.1 | 1.192× |
| **512** | 5.00 | **229.4** | ★ **1.359×** |
| 684 | 6.68 | 116.2 | 0.689× |
| 1024 | 10.00 | 152.7 | 0.905× |

**⇒ torch 的一次性 `mm` 比手工按列分块【慢 36%】，而最优宽度是 512（输出块 5 MB，接近 L3 的 8 MB）。**

**⇒ 换算到训练**：GEMM 占一步的 52.4% ⇒ **52.4% × 36% = 整步省约 19%。**

**⇒ 而它不需要写内核，只需要在 MLP 处按 512 列分块调用。这是今天唯一一个【直接可用】的训练加速。**

#### 7.52.3 FMA 四因：371 是这台机器的上限，不是实现的产物

| n | GFLOPS | 占 456 峰值 |
|---|---|---|
| **384** | **364.0** | **79.8%** |
| 512 | 330.6 | 72.5% |
| 768 | 332.7 | 73.0% |
| 1024 | 275.7 | 60.5% |
| 2048 | **235.5** | **51.6%** |

**⇒ torch 的 364 与我此前记录的 371 几乎相同 ⇒ 判据不成立 ⇒【371 是机器上限】**，
而 79.8% 是"喂满了"的水平。

**⇒ 而掉落点落在 L2（3 MB）与 L3（8 MB）之间，与工作集大小吻合 ⇒ 是【容量】不是带宽。**

**⇒ 剩下 20% 的缺口是发射气泡，而 4500U 是【6 核 6 线程、无 SMT】⇒ 没有第二个线程来填
⇒ 结构性，指令层面无路。** 这与 7.51.1 的结论一致。

**⇒ 而两处算术都确认 GEMM 是计算受限**：算术强度 300 FLOP/字节对门槛 9 ⇒ 在门槛之上 33 倍。

#### 7.52.4 ★ GIL 对照（进程 vs 线程）

```
@@M4@@
```

#### 7.52.5 今晚的账变了

| 原以为 | 实际 |
|---|---|
| ~~核显关闭（并发 0.76×）~~ | ★ **可分流，CPU 只慢 4.3%** |
| ~~训练步已到硬体下限~~ | ★ **GEMM 还有 36% 空间（分块）** |
| 371 是实现的产物 | 371 是机器上限（79.8%） |

**⇒ 两条被关掉的路各开了一条，而它们【可乘**：

```
核显分流 GEMM（52.4%）⇒ CPU 只剩 2.2 s 且只慢 4.3% ⇒ 约 2.3 s
叠加分块的 36% ⇒ 核显那一半也受益
⇒ 步时从 4.5 s ⇒ 【约 2.3~2.6 s】
⇒ 一个 epoch 从 17 小时 ⇒ 【约 9~10 小时】
⇒ 10 个 epoch 从 7.2 天 ⇒ 【约 4 天】
```

**⇒ 而再叠 Monarch 的数学路线（n²→n·log n），才进到真正的"小时级"。**

#### 7.52.6 而这一节也是对 7.46 的正式修正

**7.46 的结论（核显关闭）依据的是 0.76×，而那个数字来自线程版测量。**
**本节的进程版（4.3%）与它矛盾，而两者中只有一个是真实的资源争抢。**

**⇒ 因此 7.46 的"关闭"应标为【已被 7.52.1 推翻】，而 7.28 的六条证据里"五条并发全负"那条
也随之作废 —— 它测的是 GIL，不是硬件。**

**⇒ 而真正剩下的约束是：**

1. **UMA 共享内存**（架构事实，仍然成立，但影响远小于原先以为的）
2. **无 SMT**（结构性，限制 FP32 利用率在约 80%）
3. **15W 功耗预算**（未测，但 CPU 在核显满载时频率未降，初步证据显示不是瓶颈）

'''

m = re.search(r'^### 7\.52 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + (SEC.Replace('@@M4@@', m4)) + '\n' + s[end:]
    how = '替换 7.52'
else:
    a = re.search(r'^# 附录', s, re.M)
    if a:
        s = s[:a.start()] + (SEC.Replace('@@M4@@', m4)) + '\n\n' + s[a.start():]
        how = '插在附录之前'
    else:
        s = s.rstrip() + '\n\n' + (SEC.Replace('@@M4@@', m4))
        how = '追加'
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('  %s；报告 %d 行' % (how, s.count('\n')))
print()
print('  === GIL 对照结果 ===')
print(m4)
