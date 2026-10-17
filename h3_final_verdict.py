"""Write the final H3 verdict into the report, using the numbers that were actually measured.

The unattended chain's last link failed, and it is worth recording why: h3_verdict.txt was
written by PowerShell's `*>` redirection, which encodes with the console codepage, so every
non-ASCII character came out as mojibake. My parser matched on '≤' and therefore found
nothing and refused to write -- correctly, since it is built to refuse rather than guess.

The numbers survived as ASCII, and they are read here from the analysis output directly.

    steps to target, smoothed window 50:
        6.2   122 vs 165   +35.2%
        6.0   165 vs 185   +12.1%
        5.7   326 vs 329    +0.9%
        5.6   335 vs 340    +1.5%
        5.5   566 vs 571    +0.9%
    final at 2000 steps:  adamw 5.1241 · muon(NS=5) 4.7782 · muon3(NS=3) 4.8090
    optimiser:            4443.0 -> 3536.9 ms/step, saving 906.1 ms (20.4%)
    this harness's whole step (measured):  8.253 -> 7.278 s, 1.1339x

The question the objective asked was whether the early deficit persists. It does not: +35.2%
at loss 6.2 falls to +0.9% at loss 5.5, and the final values differ by 0.64%.

The judgement rule fixed in 7.31.4.8 is applied as written: within 5% at loss <= 5.5 counts
as equivalent. +0.9% is inside it, so the verdict is continue.

Three things are labelled rather than blended, as the plan requires:
  * the microbenchmark's NS-local figure (0.61x over the whole parameter set);
  * the optimiser stage, measured in the A/B (20.4%);
  * this harness's whole step, measured by the split (1.1339x) -- and the split's own caveat
    that the optimiser is 31.8% of a harness step, a ratio that does not transfer to the
    trainer, whose step also contains verification every 250 steps.
The 1.24x arithmetic ceiling from the microbenchmark is not quoted as a result anywhere.
"""
from __future__ import annotations

import io
import os
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''#### 7.31.4.9 2000 步验证结果（__DATE__）

**结论：继续。** 判据按 7.31.4.8 事先约定的机械应用——在 loss ≤ 5.5 上相差 0.9%，
落在 5% 等价带内。

##### 到同一 loss 所需的步数（平滑窗口 50）

| 目标 loss | NS=5 | NS=3 | 差 |
|---|---|---|---|
| 6.2 | 122 | 165 | **+35.2%** |
| 6.0 | 165 | 185 | +12.1% |
| 5.9 | 179 | 197 | +10.1% |
| 5.8 | 190 | 323 | +70.0%（该点噪声大） |
| **5.7** | **326** | **329** | **+0.9%** |
| **5.6** | **335** | **340** | **+1.5%** |
| **5.5** | **566** | **571** | **+0.9%** |

**⇒ 目标问题（早期落后是否随训练长度消失）的答案：会。** 在 loss 6.2 处 NS=3 落后
**35.2%**，到 loss 5.5 处只剩 **0.9%**。600 步时观察到的 +12% 不是持续差距，而是暂态。

##### 2000 步末值

| 臂 | 末值 | 相对 NS=5 |
|---|---|---|
| AdamW8bit | 5.1241 | — |
| **Muon（NS=5）** | **4.7782** | 基线 |
| **Muon（NS=3）** | **4.8090** | **+0.64%** |

##### 三种口径（分开，不混用）

| 口径 | 数值 | 性质 |
|---|---|---|
| NS 局部 | 1945 → 1182 ms（0.61×） | **微基准** |
| **优化器阶段** | **4443.0 → 3536.9 ms/步 ⇒ 省 906.1 ms（20.4%）** | **A/B 实测** |
| **本 harness 的完整一步** | **8.253 → 7.278 s ⇒ 1.1339×** | **分解实测** |
| 完整一步（算术上限） | 3.952 → 3.189 s ⇒ 1.24× | ★ **上限，不作为结果** |

**本 harness 的分解同时给出一个不可外推的比例**：优化器占该 harness 一步的 **31.8%**。
**它不能搬到训练器上**——训练器每 250 步做一次验证，其 3.952 s/步 是另一口径。

##### 时间口径下的净收益（本 harness）

以 2000 步、NS=5 为基线：
* 优化器省 906.1 ms/步 ⇒ 2000 步省 **1812 s ≈ 30 分钟**；
* 整步 8.253 → 7.278 s ⇒ 2000 步从 **4.58 小时** 降到 **4.04 小时**，即 **省 0.54 小时**。

**这是本 harness 的数字，不是训练器的**；训练器的净收益需要它自己的分项，
**该测量未做**，因此不作推断。

##### 自动化链的一个失败（如实记录）

收口的最后一环**没有自动完成**：`h3_verdict.txt` 由 PowerShell 的 `*>` 重定向写出，
使用控制台代码页，**所有非 ASCII 字符变成乱码**；解析器按 `≤` 匹配因而找不到任何数字，
于是**拒绝写入**——这是设计行为（宁可不写也不猜），但意味着链条断在最后一环，
本节由人工从分析输出中读取后写入。

同时 `ns_h3_fwd_bwd_split.py` 第一次运行**崩溃**（`a leaf Variable that requires grad is
being used in an in-place operation`），因为参数更新没有包在 `torch.no_grad()` 里；
它的护栏此前从未让它在训练结束前跑过，所以这个 bug 一直没暴露。**已修复并重跑。**

**⇒ 两处都说明同一件事：无人值守的链必须逐环干跑，否则断点只会出现在没人看的时候。**

##### 仍未建立的部分

1. **只有一个随机种子**（计划要求 ≥3）；
2. **没有生成质量评估**，只有验证 loss；
3. **训练器的分项未做**，所以本 harness 的 1.1339× 不能直接换算成训练器的加速；
4. `train_ar_v2.py` **没有 Muon 路径**，Muon 仍只是 `ab_muon_v2.py` 里的候选。

##### 只推荐一个下一步

**把 Muon + NS=3 接进 `train_ar_v2.py` 的训练路径，再跑一次多种子验证。**

理由：两个杠杆现在都有实测支撑（Muon 时间口径 1.96×、NS 3 在 Muon 上再省 20.4%，
整步实测 1.1339×），但它们**都还没进入正式训练器**——`train_ar_v2.py` 里既没有 Muon，
也没有 NS。在把加速写进任何预期的训练时长之前，这一步是必须的。
'''

SEC = SEC.replace('__DATE__', time.strftime('%Y-%m-%d %H:%M'))
m = re.search(r'^#### 7\.31\.5 ', s, re.M)
if m:
    s = s[:m.start()] + SEC + '\n' + s[m.start():]
    how = '插在 7.31.5 之前'
else:
    s = s.rstrip() + '\n\n' + SEC
    how = '追加在末尾'
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('  %s；报告 %d 行' % (how, s.count('\n')))
