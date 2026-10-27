"""Append 7.53: the contradiction resolved. The iGPU offload is viable.

Three measurements of the same thing now tell one story. The duty-cycle sweep puts the CPU slowdown
between 10% and 19% across the whole range and almost flat, which means neither of the two extremes
was typical: 4.3% came from a bursty low-intensity measurement and 94.4% did not reproduce at all.
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

SEC = '''### 7.53 矛盾解开：核显卸载【可行】，而判据通过

#### 7.53.1 占空比扫描

| 核显占空比 | CPU s/步 | 相对基线 | 核显 GFLOPS |
|---|---|---|---|
| 0% | 3.189 | 1.000× | — |
| 25% | 3.677 | 1.153× | 139.0 |
| 50% | 3.541 | 1.110× | 142.3 |
| **75%** | 3.504 | **1.099×** | 169.9 |
| **100%** | 3.808 | **1.194×** | 409.2 |

**判据（7.52.3 事先固定：≥75% 占空比时拖慢 > 25% ⇒ 不可行）⇒ 19.4% < 25% ⇒ ★【通过】。**

#### 7.53.2 而三次测量现在是一个故事

| 测量 | 拖慢 | 性质 |
|---|---|---|
| ① 任务分流 | 4.3% | 短促低强度测量（506.9 GFLOPS 含组间启动开销） |
| ④ 进程版 | **94.4%** | ★ **本次未能复现 ⇒ 异常** |
| **★ 本次扫描** | **10~19%** | **曲线【平坦】，与占空比几乎无关 ⇒ 真实值** |

**⇒ 而"平坦"这件事本身是结论**：拖慢不是"核显越忙越严重"，而是**只要核显在工作，
就会有一个约 10~20% 的固定代价**。这与"共享 DDR4 的固定争抢"一致，
而与"随强度增长的拥塞"不一致。

**⇒ 所以 ④ 的 94.4% 最可能是那次测量的偶发异常（例如另一个进程同时在跑），
而 ① 的 4.3% 是短促测量低估了稳态代价。**

#### 7.53.3 收益计算

```
一步 = 4.53 s = GEMM 2.36 s（52.4%）+ 其余 2.14 s（47.6%）

核显接走 GEMM:
  iGPU 409.2 GFLOPS 对 CPU 的 240 ⇒ 2.36 s × (240/409.2) = 1.38 s
  CPU 只剩 2.14 s,但被拖慢 1.19× ⇒ 2.55 s
  两者重叠 ⇒ max(2.55, 1.38) = 【2.55 s】

⇒ ★【相对 4.53 s = 1.78×】

再叠 L2 分块（7.52.1 已证 1.359×,作用于 GEMM 那部分）
  核显那一半也受益 ⇒ 1.38 s → 约 1.02 s
  CPU 的非 GEMM 部分不变 ⇒ 2.55 s
⇒ 约 【2.55 s → 仍受 CPU 那半支配】

⇒ ★【总计约 1.78×,而天花板由"CPU 剩下那 47.6%"决定】
⇒ 一个 epoch 从 17 小时 ⇒ 【约 9.6 小时】
⇒ 10 个 epoch 从 7.2 天 ⇒ 【约 4 天】
```

**⇒ 而关键的结构性认识**：**卸载 GEMM 之后，瓶颈从 GEMM 转移到 attention + elementwise
（CPU 的那 47.6%）。** **⇒ 所以下一步的杠杆不再是 GEMM，而是 attention 的 18.3% 与
elementwise 的 19%。** 而 elementwise 已经融合、attention 已是最优（7.47）⇒ **CPU 侧的这 47.6%
基本没有压缩空间。**

**⇒ 因此 1.78× 接近于"纯工程手段"的上限。再往上只能靠数学（7.51 的 Monarch）。**

#### 7.53.4 对 7.46 与 7.52.3 的最终修正

| 版本 | 结论 | 状态 |
|---|---|---|
| **7.46** | 核显关闭（0.76×） | ✗ **推翻** —— 那个数字来自线程版且未复现 |
| **7.52.1** | 核显重开（4.3%） | △ **方向对但数字低估** |
| **7.53** | **核显可卸载，代价 10~19%，净收益 1.78×** | ✓ **本次结论** |

**⇒ 而真正的约束仍是 UMA 共享内存，只是它的代价是【固定的 10~20%】而不是【致命的】。**

#### 7.53.5 这条纪律的第十三次触发

**今天同一个问题测了三次，得到 4.3% / 94.4% / 10~19%。**

**⇒ 而如果我只做了其中一次，就会得出三个完全不同的结论：**
**"核显可用"（①）· "核显废了"（④）· "核显有代价但值"（③）。**

**⇒ 而三次里，只有把【自变量显式扫出来】的那次给出了可解释的曲线。**
**前两次都是单点，而单点无法区分"这个现象的规律"与"这一次的偶然"。**

**⇒ 所以纪律再进一步：当两次测量矛盾时，正确做法不是重测一次取平均，
而是【把导致矛盾的变量找出来并扫它】。** 那样得到的不是"更可信的单点"，
而是【一条能解释两个旧点的曲线】。

'''

m = re.search(r'^### 7\.53 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.53'
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
