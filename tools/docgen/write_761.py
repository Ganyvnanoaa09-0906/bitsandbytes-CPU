"""Append 7.61: the iGPU question is settled. The device itself is slowed by CPU activity.

The phase split separates enqueue, device and sync, and the answer is unambiguous. With the CPU idle
the kernel takes 14.7 ms and the phases add cleanly: 0.316 ms to enqueue, 14.374 ms to synchronise.
With a 30.5 ms CPU burst inserted after the launch, the synchronisation returns 28.97 ms -- the kernel
needed only 14.4 ms, so it should have completed during the burst and finish should have been
immediate. It was not, which means the device had not finished. Total under load, about 59 ms against
14.7 idle, a factor of four.

So it is not the host path. Enqueuing costs 0.18 ms regardless of load, and the keep-alive idea cannot
help because the problem is not the device idling. Async does not help either: the device genuinely
needs the longer time, so overlapping just moves the wait.

Bandwidth is not the cause: this call moves 32 MB in 14.4 ms, so 2.2 GB/s, and the CPU side about 1
GB/s, against 27 available. Twelve percent of the bus.

What remains is the 15 W shared power budget and the 8 MB shared L3, and both mean the same thing --
this is hardware interference, not scheduling, and during training the CPU is never idle.
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

SEC = '''### 7.61 核显定案：设备本身被 CPU 活动拖慢，而不是 host 路径

#### 7.61.1 三段分离

| 条件 | 入队 ms | 设备忙 ms | **同步 ms** | 合计 ms |
|---|---|---|---|---|
| **紧循环（CPU 空闲）** | — | — | — | **14.717**（389.9 GFLOPS） |
| 三段分离（CPU 空闲） | 0.316 | — | **14.374** | 14.690 |
| **CPU 在内核执行期间忙 30.5 ms** | 0.181 | 30.5 | ★ **28.970** | **59.655** |
| 对照：先同步再让 CPU 忙 | 0.168 | — | **30.419** | — |

#### 7.61.2 而第 3 组是决定性的

```
启动后 CPU 忙了 30.5 ms,而设备只需要 14.4 ms
⇒ 同步【本该】返回约 0
⇒ ★【而它返回了 28.970 ms ⇒ 设备在那 30.5 ms 里【没有算完】,又花了 29 ms】
⇒ 负载下合计约 59 ms 对空闲 14.7 ms ⇒ 【慢 4 倍】
```

**⇒ 所以不是 host 侧。入队即使 6 个 torch 线程占满 CPU 也只要 0.181 ms。**

#### 7.61.3 带宽也不是原因

```
本次调用搬运 32 MB（读 A/B + 写 C）/ 14.4 ms ⇒ 【2.2 GB/s】
CPU 的 torch.mm 190 GFLOPS ⇒ 约 【1 GB/s】
合计 3.2 GB/s,而实测总带宽 27 GB/s ⇒ 【用了 12%】
```

#### 7.61.4 剩下两个候选，而它们指向同一件事

| 候选 | 证据 |
|---|---|
| **★ 15W 共享功耗预算** | 6 个 CPU 线程本身就吃满 15W，再加核显 ⇒ **双方一起降频** |
| **L3 争抢** | 8 MB 共享，而 CPU 的 GEMM 在猛刷它 |

**⇒ 两者都是【硬件层面的互相干扰】，不是调度问题。**

#### 7.61.5 三个方案各输在哪

| 方案 | 为什么不行 |
|---|---|
| **keep-alive 假任务** | 问题不是"GPU 闲置"，是**"CPU 一忙 GPU 就慢"** ⇒ 保活不解决 |
| **异步 / 不立刻同步** | **设备本身需要 59 ms（不是 14.4）** ⇒ 异步只是让它慢慢跑，总时间不变 |
| **给驱动留一个核** | host 路径本就不是瓶颈（入队 0.18 ms）⇒ 留核没用 |

**⇒ 而这也一次解释了今天所有的核显异常：**

| 现象 | 真实原因 |
|---|---|
| 并发 0.76× | CPU 忙 ⇒ 核显慢 2~4 倍 ⇒ 合计反而下降 |
| 探针 2~5 倍慢 | 同上（探针里 CPU 刚跑完 torch.mm） |
| 409 对 90 GFLOPS | 紧循环（CPU 闲）对 CPU 忙 |
| **7.53 的 1.78×** | **建立在"CPU 闲时的核显速度"上,而训练里 CPU 从不闲** |
| **33446 GFLOPS** | **入队时间被当成了执行时间**（7.60.2 的同一次误读） |

#### 7.61.6 最终结论

**⇒ 核显这条路【从根上不通】—— 不是代码问题，而是"这台 15W 的 APU 里,
两个引擎不能同时忙"这个硬件事实。**

**⇒ 而 7.46 最初的"关闭"是对的，只是理由要改成：不是"抢内存带宽"，
是【抢同一个 15W 预算和同一个 8 MB L3】。**

**⇒ 而这条与"库"这个身份也有关**：**混合 CPU/GPU 执行在任何共享功耗预算或末级缓存的
设备上都有这个问题** —— 而消费级 APU、以及所有在同一个封装里放 CPU 和 GPU 的设计都属此类。

'''

m = re.search(r'^### 7\.61 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.61'
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
