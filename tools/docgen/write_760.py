"""Append 7.60: the task probe overturns the iGPU conclusion, and 1.78x is withdrawn.

The probe measured every operation of the training step on both engines, wall-clock per call, both
sides measured the same way. The iGPU lost all seven, by 2x to 5x. That contradicts the 409 GFLOPS
the duty-cycle sweep reported, and the reason is in that sweep's own accounting: it accumulated only
the kernel time and divided by that, excluding the queue.finish() synchronisation, while the CPU side
of the same comparison was wall clock. Two engines compared with two different denominators.

So 7.53's 1.78x is withdrawn, and 7.46's closure turns out to have been right with the wrong reason:
not shared-memory contention, but the iGPU simply being slower on these shapes.
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

SEC = '''### 7.60 按任务分派探针：核显【全输】，而 1.78× 撤回

#### 7.60.1 探针结果

**每个操作都在两个引擎上测，口径相同（`perf_counter` 包住整次调用，含同步）：**

| 操作 | 形状 MxNxK | CPU ms | 核显 ms | **核显/CPU** | 归属 |
|---|---|---|---|---|---|
| qkv 投影 | 2048×1536×512 | 18.588 | 49.708 | **0.37×** | CPU |
| attention 分数 | 1024×1024×64 | 1.532 | 4.068 | **0.38×** | CPU |
| attention 输出 | 1024×64×1024 | 0.715 | 3.557 | **0.20×** | CPU |
| mlp w12 | 2048×2736×512 | 31.863 | 63.452 | **0.50×** | CPU |
| mlp w3 | 2048×512×2048 | 31.185 | 73.199 | **0.43×** | CPU |
| 优化器 NS（小） | 512×512×512 | 2.431 | 5.618 | **0.43×** | CPU |
| 优化器 NS（大） | 2048×512×512 | 7.008 | 19.579 | **0.36×** | CPU |

**⇒ ★【核显赢下 0 / 7 个操作，而每个都慢 2~5 倍。】**

**⇒ mlp w12：CPU 180 GFLOPS · 核显【90 GFLOPS】。**

#### 7.60.2 那 409 GFLOPS 是怎么来的 —— 分母不同

```python
on += time.perf_counter() - ts        # 只累加【跑一组】的时间
grp()                                  # 而 grp() 里【含 cq.finish()】
queue.put((n, on, el, 2.0*FLOP*n/on/1e9))   # ★ GFLOPS 用 on 算
```

**⇒ 所以 7.53 的 409 GFLOPS 是【内核执行时间】口径，而墙钟里还有 `queue.finish()` 的同步等待，
而它在这里占大头。**

**⇒ 而同一段代码里 CPU 那侧用的是【墙钟】⇒ ★【两个引擎用了两个不同的分母。】**

**⇒ 这是第十六次同形错误，而它在我自己的测量里**：GFLOPS 这个判据没错，
**但它除的不是"两个引擎共用的那个时间"。**

#### 7.60.3 于是结论要改回来

| 版本 | 结论 | 状态 |
|---|---|---|
| **7.46** | 核显关闭（0.76×）—— 理由：共享内存争抢 | △ **结论对，理由错** |
| **7.53** | 核显可卸载，1.78× | ✗ **撤回**（数字虚高） |
| **7.60** | **核显在真实形状上比 CPU 慢 2~5 倍** | ✓ **本次结论** |

**⇒ 7.46 的"关闭"其实是对的，只是它把原因归给"共享 DDR4 争抢"，而真相是【核显在这些形状上
就是慢】—— 与争抢无关。**

**⇒ 于是"CPU 240 + 核显 361.5 = 601.5 = 一张 1030"这个等式【不成立**：
那个 361.5 是内核时间口径，而真实墙钟下核显只有约 90 GFLOPS。

**⇒ ★【天花板回到 CPU 单干。】**

#### 7.60.4 而三条路的账最终定格

| 路线 | 结论 |
|---|---|
| 指令层面 | ✗ 无 SMT，结构性死角 |
| **工程（核显卸载）** | ✗ **撤回 —— 核显真实墙钟比 CPU 慢 2~5×** |
| 数学（Monarch） | ✗ FLOPs 降 10~28× 而壁钟只 0.81~1.30× |
| **⇒ 结果** | **★【没有可用的加速，除了库层面的静默回退修复】** |

**⇒ 而唯一剩下的、且已证实的是一个数量级的活，在【库】那一侧**：
`gemv_4bit_inference` 只有 bf16 ⇒ AVX2-only 机器静默回退 ⇒ **我们填了，10.6×**。

#### 7.60.5 而这条纪律的第十六个实例值得单独写出来

> **判据（GFLOPS）本身没错，但它除的是【内核时间】而不是【两个引擎共用的墙钟时间】。**
> **⇒ 跨引擎比较时，分母必须是同一种时间 —— 而"更精细的口径"在比较里反而是错的。**

**⇒ 而它为什么没被发现**：409 GFLOPS 是个【好看的数字】，而好看的数字最容易被放过 ——
**与 7.54 里那个 1.359× 完全一样的心理机制。**

'''

m = re.search(r'^### 7\.60 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.60'
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
