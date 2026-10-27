"""Append 7.46: the iGPU is closed, now on correct evidence with a fast kernel.

This is the strongest form the conclusion can take. All five earlier concurrency experiments used a
kernel that could not beat the CPU, so they proved only that handing work to a slower device does
not help. This one uses the swizzled kernel at 362.6 GFLOPS, which is 1.34x the CPU's 269.8, and
concurrency still comes out at 204.2 -- 0.76x of the CPU alone, and 32% of the 632.4 that perfect
stacking would give.

So the reason the iGPU cannot be used is not that its kernel is slow. It is that the two engines
share one DDR4 bus. Which is precisely what 7.40.2 predicted when the technique list came in:
Unified Memory is the disease here, not a cure, and the same feature that is a convenience on a
discrete card is the thing that rules out concurrency on an APU.
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

SEC = '''### 7.46 核显的最终结论：关闭（而这次是在正确证据上）

#### 7.46.1 为什么这次的重测才有决定性

**此前五条并发实验全部用的是跑不过 CPU 的内核**，所以它们只证明了"把工作交给更慢的设备
没有好处"——**那是平凡的结论**。7.45 修好地址映射之后，核显单内核到 362.6 GFLOPS，
**比 CPU 快 1.34 倍**，于是这笔账才真正值得算。

#### 7.46.2 结果

内核经 numpy 逐元素校验：**✓ 全对**。形状是模型的真实形状。

| 项 | GFLOPS | 相对 CPU 单独 |
|---|---|---|
| **CPU 单独**（6 线程） | **269.8** | 1.00× |
| **核显 单独**（swizzle 内核） | **362.6** | 1.34× |
| ★ **并发总和**（真并行：CPU 主线程 + 核显工作线程） | **204.2** | **0.76×** ✗ |
| 理想叠加上限 | 632.4 | 2.34× |

**⇒ 判据（7.39.4 事先固定：并发 ≥ CPU 单独 1.2 倍）【未达】**
**⇒ 实际只拿到理想叠加的 32%。**

#### 7.46.3 而这次结论的性质不同

**核心事实：即使核显的内核比 CPU 快 1.34 倍，两个一起跑仍然比 CPU 单独慢 24%。**

**⇒ 所以核显不能用的原因【不是内核慢】，而是两个引擎共享同一条 DDR4。**

**⇒ 这恰好定量证实了 7.40.2 的判断**：在那份 GPU 技术清单里，**Unified Memory 被标为"病根，
不是药方"**——在独显上它是便利，**在 APU 上它是禁用并发的根本原因**。

#### 7.46.4 7.28 的六条证据，最终状态

| 证据 | 原始状态 | 最终状态 |
|---|---|---|
| 单内核速度 | 300 GFLOPS（CPU 的 1.51×） | **362.6 GFLOPS（CPU 的 1.34×）**——修正后仍成立 |
| LDS 缺口 | 8 倍（66 对 544） | **最多 2 倍，且 swizzle 可消除**——修正 |
| **五条并发全负** | 用慢内核 | ★ **用快内核重测，仍是 0.76×**——**结论不变而证据更强** |
| occupancy 翻倍 +3% | — | 不变（也与 7.45 的"分块无益"一致） |
| 小矩阵（NS）慢 1.55× | — | 不变 |
| **UMA 架构** | 定性理由 | ★ **定量证实：并发只拿到理想叠加的 32%** |

**⇒ 六条全部指向同一个结论，而其中三条被修正后【更强】而不是更弱。**

#### 7.46.5 方法论：又一次"检查的东西没跑过"

写并发脚本时，正确性检查被放在了**内核第一次启动之前**，于是它读的是**没写过的缓冲区**，
报了一个假的失败。

**这与 7.41.6 记的第五次是同一种形状**：`np.abs(out).mean() > 0` 这个判断本身没错，
**但它检查的对象还不存在。**

**⇒ 第六次。纪律再加一条：任何检查都要确认【被检查的对象已经存在或已经运行过】。**

（修正后重跑：✓ 全对，而数值与 7.45 的 361.5 一致，说明结论未受影响。但如果不查，
我会带着一个"内核算错了"的错误印象去写报告。）

#### 7.46.6 那么核显这条线还剩什么

**只剩一件事：单跑。** 362.6 对 269.8 ⇒ **若整个 GEMM 都交给核显，那部分快 1.34 倍。**
而 GEMM 占一步的 52% ⇒ 理论上限 **52% × (1 − 1/1.34) = 13%**。

**但前提是 CPU 在那 52% 的时间里完全空闲**，而这正是实际训练做不到的——训练的其余部分
（attention、elementwise、优化器）都在 CPU 上，**它们一跑就抢内存**，而 7.46.2 已经测出
抢的代价是 24%。

**⇒ 除非把整个训练都搬到核显上，否则这条路是死的。而核显没有独立显存、只有 6 个 CU，
跑不了这个模型。**

**⇒ 核显问题到此关闭。**

'''

m = re.search(r'^### 7\.46 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.46'
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
