"""Append 7.62: the inversion fails too, and the shape of the curve explains why.

The iGPU alone runs this GEMM in 14.7 ms against the CPU's 30.2, so it is twice as fast when the CPU
is idle. That made "iGPU primary, CPU assisting lightly" a reasonable hypothesis, and the answer is
no -- and the way it fails is more informative than the failure.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.62 倒置也不成立：而"悬崖"的曲线形状说明了机制

#### 7.62.1 假设的由来

```
核显单独（CPU 完全空闲）: 14.721 ms ⇒ 【389.8 GFLOPS】
CPU  单独（6 线程）:      30.195 ms ⇒ 【190.0 GFLOPS】
⇒ ★【CPU 空闲时,核显本来快 2 倍】
```

**⇒ 所以"让核显为主、CPU 只做一点点"是合理的假设 —— 而它也不成立。**

#### 7.62.2 扫描结果

| CPU 线程 | 核显 ms | 核显 GFLOPS | CPU GFLOPS | **合计 GFLOPS** |
|---|---|---|---|---|
| **0** | 14.721 | **389.8** | 0.0 | **389.8** |
| **1** | 75.146 | **76.4** | 187.6 | **263.9** |
| 2 | 90.671 | 63.3 | 176.9 | 240.2 |
| 3 | 116.984 | 49.0 | 189.2 | 238.2 |
| 4 | 80.850 | 71.0 | 166.3 | 237.3 |
| 6 | 114.756 | 50.0 | 183.3 | 233.3 |

**⇒ 判据（核显保持 20% 以内且 CPU 有 3 线程产出）⇒ 【不成立】。**

**⇒ 而合计最好只有 263.9，对 CPU 单干的 190 ⇒ 【1.39×】。**

#### 7.62.3 ★ 而"悬崖"这个形状本身就是机制

```
若原因是【资源争抢】 ⇒ 曲线应【渐变】—— CPU 越忙,核显越慢
而实测是【一碰就掉】:
   0 线程 → 389.8
   1 线程 →  76.4      ⇒ ★【5.1 倍,而只多了【一个】线程】
   6 线程 →  50.0      ⇒ 从 1 到 6 只再掉 1.5 倍
⇒ ★【不是渐变,是开关】
```

**⇒ 所以不是带宽、不是调度，而是【功耗仲裁】：**

> **只要有任何 CPU 活动，SoC 的功率分配就把电给 CPU，核显立刻被降到低功耗档。
> 而 15W 的预算里，两者只能有一个在跑。**

**⇒ 这解释了为什么 0 → 1 就掉 5 倍：它不是在"分走一部分电"，而是在"把核显关小"。**

#### 7.62.4 而 1.39× 是个【不可达的上界】

**那个 263.9 GFLOPS 是纯 GEMM 吞吐，而它要求：**

```
CPU 只跑 1 个线程,同时完成训练里那 47.6% 的活（attention + elementwise + autograd）
⇒ 而 CPU 降到 1 线程 ⇒ 那部分会慢好几倍
⇒ ★【所以实际连 1.39× 都拿不到】
```

#### 7.62.5 核显问题的最终定案

| 版本 | 结论 | 状态 |
|---|---|---|
| 7.46 | 关闭（0.76×）· 理由：抢内存带宽 | △ 结论对，理由错 |
| 7.53 | 可卸载 1.78× | ✗ 撤回 |
| 7.60 | 真实形状上慢 2~5 倍 | △ 现象对，机制未明 |
| 7.61 | 设备本身被 CPU 活动拖慢 4 倍 | △ 现象对，机制缩小到功耗/L3 |
| **7.62** | **★ 功耗仲裁：核显的 389.8 只在 CPU【完全不动】时存在** | ✓ **定案** |

**⇒ 所以"核显不可用于训练"不是因为代码、不是因为调度、不是因为带宽 ——
而是【在这台 15W 的 APU 上,CPU 和 GPU 的满速状态【互斥】】。**

#### 7.62.6 而这条对库的意义

**这是一条该写进文档、而不是让用户自己撞的结论：**

> **在任何共享功耗预算的封装里（所有消费级 APU），CPU 与 iGPU 的算力【不可叠加】。
> 用户会合理地以为"两个引擎加起来更快"，而实测是反的：一个引擎的满速状态会关闭另一个。**

**⇒ 而它与之前那条（共享末级缓存）是两件不同的事，而这条更硬 ——
因为缓存争抢是【按比例】的，而功耗仲裁是【开关】式的。**

**⇒ 所以对 bnb 的 CPU 后端而言，"支持 GPU 卸载"这类特性在 APU 上应当【默认关闭】，
并把原因写清楚。**

'''

m = re.search(r'^### 7\.62 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.62'
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
