"""Append 7.43: the iGPU kernel's bug found and fixed, and the trustworthy number.

The cause of the wrong results was visible in the source once I looked for it: the staging loops
wrote four times out of bounds.

    // stage A: TS*TK = 1024 floats = 256 float4
    for (int p = 0; p < 4; p++) {
        int f4 = p * 256 + lane;     // lane is 0..255, so f4 reaches 1023
        int r = (f4 * 4) / TK;       // r reaches 255
        As[r][c] = ...;              // As is [64][20]
    }

The tile is 1024 floats = 256 float4 with 256 threads, so it is one iteration, not four. The
excess writes land past As, on Bs, which is declared next to it -- which is why the damage was
0.03% of elements rather than a crash.

After the fix every shape is correct and throughput rises about 10%, because the stray writes were
also damaging the local memory layout.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.43 核显内核的 bug 找到并修好：真实数字是 170 GFLOPS

#### 7.43.1 原因：staging 循环越界 4 倍

7.41.4 测出核显 GEMM 有 0.03%~0.15% 的元素错误。去看源码，原因一眼可见：

```c
// stage A: TS*TK = 1024 floats = 256 float4      ← 注释说 256
for (int p = 0; p < 4; p++) {
    int f4 = p * 256 + lane;                       // lane 是 0..255 ⇒ f4 到 1023
    int r = (f4 * 4) / TK;                         // ⇒ r 到 255
    ...
    As[r][c] = ...;                                // 而 As 是 [64][20]
}
```

**tile 是 1024 个 float（= 256 个 float4），线程也是 256 个 ⇒ 每个线程一次，不是四次。**
多出来的写入越过 `As` 的末尾，落到紧挨着声明的 `Bs` 上——**这正好解释为什么是 0.03% 而不是
崩溃**：只有少数 B 值被污染。

**修法**：把 `f4 = p * 256 + lane` 的循环换成单次 `f4 = lane`（stage A 与 stage B 各一处）。

#### 7.43.2 修好之后：全部正确，而且更快

| 形状 | 修前（有错） | **修后** | 相对 CPU |
|---|---|---|---|
| 1024×2736×512 | 155.5 ✗ | **170.5 ✓ 全对** | 0.86× |
| 1024×512×2736 | 152.9 ✗ | **168.1 ✓ 全对** | 0.85× |
| 2048×2048×2048 | 163.1 ✗ | **169.5 ✓ 全对** | 0.85× |
| 1024×512×512 | 128.2 ✗ | **131.9 ✓ 全对** | 0.66× |

**错误全部归零，而吞吐反而涨了约 10%。** 原因：那些越界写入不只污染结果，**还破坏了本地内存
的布局**（`Bs` 被反复踩），所以修 bug 同时是性能修复。

#### 7.43.3 核显的可信数字

| 说法 | 数值 | 状态 |
|---|---|---|
| 报告曾记 | **300 GFLOPS** | ✗ **从来不存在**（7.28.5 与 7.41.4 两次确认） |
| 修前实测 | 155.5（带错） | ✗ 无效 |
| **修后实测** | **约 170 GFLOPS，全部正确** | ✓ **可信** |
| CPU 同形状 | 218~240 GFLOPS | 核显的 **1.16×** |

**⇒ 核显在本机真实形状上比 CPU 慢，约 0.86×。** 而 7.28 记的"300 GFLOPS，CPU 的 1.51 倍"
**完全作废**——那是带 bug 的内核加上不可复现的版本。

#### 7.43.4 而基准现在可信了，所以分块实验才有意义

**这是本节真正的价值**：在 7.41.4 的状态下测任何优化都是在错误的地基上盖楼——
一个算出 99.97% 正确结果的核显**不能用来衡量分块或布局的收益**。

**⇒ 现在可以开始抄 Swizzle 与寄存器分块了**，判据仍然是事先写好的：单内核 ≥ 400 GFLOPS
⇒ 说明 LDS 不是硬墙、7.28 的结论要改；否则确认 LDS 是硬墙。

#### 7.43.5 与既有结论的关系

**7.28 的六条证据里，只有"单内核速度"这一条被改动**（从 300 降到 170，且方向从"比 CPU 快
1.51×"变成"比 CPU 慢"）。**其余五条不受影响**：

* LDS 66~68 GB/s 对 544（实测，与内核速度无关）
* 五条并发实验全负（测的是并发，不是单内核）
* occupancy 翻倍只 +3%
* 小矩阵（NS）比 CPU 慢 1.55×
* **⇒ 核显不能与 CPU 并用的结论不变，而且现在理由更充分：它单独跑都比 CPU 慢。**

'''

m = re.search(r'^### 7\.43 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.43'
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
