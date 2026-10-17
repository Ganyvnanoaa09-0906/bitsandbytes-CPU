"""Append 7.44: the register-blocking experiment is blocked by an OpenCL driver failure.

Not a result, an environment failure, recorded so the next round does not mistake it for one.

The experiment itself was prepared and is worth running once the GPU works again: the kernel's
inner loop reads broadcast (all threads sharing a ty read the same As address), and the standalone
microbenchmark measured broadcast at 271 GB/s against the 544 needed -- a 2x gap, not the 8x I
reported earlier from assuming the strided pattern applied. Register blocking at 8x8 per thread
halves the LDS reads per FMA, from 0.5 to 0.25, which is exactly the right size for a 2x gap.

The failure: after many OpenCL runs today, cl.CommandQueue now returns OUT_OF_HOST_MEMORY while
cl.Context succeeds. The minimal case -- build context, then build queue, nothing else -- fails at
the queue, and igpu_gemm_v4.py, which ran successfully twenty minutes earlier, fails the same way.
A reboot is the usual remedy and has not been attempted because the machine is in use.
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

SEC = '''### 7.44 寄存器分块：实验已就绪，但被 OpenCL 驱动故障挡住

**这不是结果，是环境故障，记录下来以免下一轮把它当成结果。**

#### 7.44.1 而实验本身有一个重要的口径修正

重看 v4 内核的内层读取：

```c
a[i] = As[ty*TM + i][k];      // k 固定，只随 ty 变 ⇒ 同一 ty 的所有 tx 读同一地址
b[j] = Bs[k][tx*TN + j];      // 只随 tx 变 ⇒ 同一 tx 的所有 ty 读同一地址
```

**两种读取都是【广播】模式，不是跨步模式。**

而我的 LDS 微基准测过三种模式：**连续 68.2 · 广播 271.2 · 跨步 66.0 GB/s**。

**⇒ 所以内核实际能拿到的 LDS 带宽是 271 GB/s，而不是 66~68。**
**⇒ 相对所需的 544，缺口是【2 倍】，不是我此前报的 8 倍。**

**那个 8 倍的说法（7.28.6）建立在"内核用跨步模式"的假设上，而假设是错的。**
这是同类错误的第五次——**测量自身没错，但我把另一份测量的结论套到了它头上。**

#### 7.44.2 而 2 倍的缺口正好是寄存器分块的量级

| 每线程 tile | 每 K 步的 LDS 读 | 每次读服务的 FMA | 读/FMA |
|---|---|---|---|
| 4×4（现状） | TM+TN = 8 | 2 | **0.50** |
| **8×8** | TM+TN = 16 | 4 | **0.25** |

**读量减半，总计算不变。** 代价是 64 个累加器而非 16 个——GCN 每线程 256 个 VGPR，应该装得下，
但占用率会降；而占用率已被 7.28 排除为杠杆，这**恰好一致**：**本次改的是读取【总量**，
不是延迟隐藏。

**判据（7.39.4 事先固定）：单内核 ≥ 400 GFLOPS ⇒ LDS 不是硬墙，7.28 的结论要改。**

#### 7.44.3 故障现象

```
igpu_gemm_v4.py（二十分钟前成功，给出 170.5 GFLOPS 全对）⇒ 【现在也失败】
最小复现: cl.Context 成功 · 【cl.CommandQueue 抛 OUT_OF_HOST_MEMORY】✗
机器空闲内存 5.9 GB ⇒ 不是系统内存问题
```

**⇒ 上下文能建、队列建不了 ⇒ AMD OpenCL 驱动的资源状态出了问题。**
今天在核显上反复创建上下文与队列（十余次），**驱动大概是泄漏了**。重启通常能解决，
但**机器在被使用，没有重启**。

#### 7.44.4 脚本已就绪

`igpu_register_block.py` 写好了，扫描五种线程×tile 组合，参数化到
`TS = NTX*TM = NTY*TN` 自动成立、staging 迭代数由宏推导（**7.43 那个 bug 就来自硬编码它们**）。
**驱动恢复后直接跑即可。**

#### 7.44.5 顺便：一条与核显无关的收获

**广播模式 271 GB/s 这个数字，7.28.6 测出来时就摆在结果里**，而我在写结论时用了跨步模式的
66——**因为我想当然地认为"跨步才是真实模式"**。

**⇒ 与 7.41.6 的第四次是同一条纪律：结论要用【实际发生的】那个模式/路径的数字，
而不是相邻的、看起来更合理的那一个。**

'''

m = re.search(r'^### 7\.44 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.44'
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
