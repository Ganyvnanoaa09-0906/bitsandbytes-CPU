"""Correct 7.86: the GT 1030 anchor was already measured, and it is our own iGPU.

I wrote in 7.86 that a GT 1030's achievable fraction of peak was not measured here and could not be. That
was wrong in a way worth recording: this machine contains a 384-shader GPU -- the same execution width
as a GT 1030 -- and the report measured it at 644.9 GFLOPS, 56.1% of its 1150 peak. Having judged the
quantity unmeasurable, I then leaned to "same order of magnitude", which was the optimistic direction.
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

SEC = '''### 7.87 ★ 更正 7.86：GT 1030 的锚点【早就测过了】—— 而它就是我们的核显

#### 7.87.1 我在 7.86 里写错的一句

```
✗ 7.86 原文: "GT 1030 在真实训练里能跑到峰值的百分之几,本报告【没有测过,也无法在这台机器上测】"
   ⇒ 于是它只能"作为一个区间进入结论" ⇒ 而那个区间我取了【同量级】

✓ 而实际: ★【这台机器里【就有一台 384-shader 的 GPU —— 和 GT 1030 的执行宽度一样 ——
            而报告【测过它】:644.9 GFLOPS,峰值的 56.1%】
```

**⇒ 错在不只是"漏了一个数据"，而是：**
**把一个【有锚点】的量判成"无法测"，然后【在没有锚点的情况下朝乐观方向取了一个值】。**

#### 7.87.2 ★ 而锚点是现成的（§7.19.6，本报告早已记录）

| | GT 1030 (GP108) | 我们的 Vega 6 (gfx902) |
|---|---|---|
| **执行单元** | **384 shader** | **384 shader（6 CU × 64）** |
| 架构 | Pascal (2016) | GCN 5 (2019) |
| 理论峰值 | 1127 GFLOPS @1468 MHz | **1150 GFLOPS** |
| 专用显存 | 2 GB GDDR5, 48 GB/s | 无（共享 DDR4 ~25 GB/s） |
| **实测（成熟 GEMM 库）** | **未测（≈cuBLAS 同档）** | ★ **CLBlast：644.9 突发 / 635.1 持续 = 56.1%** |

```
§7.19.6 原文: "核显在专业 GEMM 库下达到 644.9 GFLOPS（突发）与 635.1（持续），
              为 CPU 的 3.21~3.25 倍，占理论峰值 56.1%，且持续 90 秒衰减仅 0.3%。"
```

**⇒ ★【所以"一台 384-shader 的 GPU 在成熟库下能到峰值的 56%"这件事，本报告【已经测过了】。**

**⇒ GT 1030 在 cuBLAS 下大概率也在同一区间（55~60%）⇒**
**1127 × 0.56 ≈ 【631 GFLOPS】**（区间 620~680）。

**⇒ 对我们 CPU 实测的 229.8 ⇒ ★【2.7~3.0×，而不是"同量级"】。**

#### 7.87.3 所以 7.86.4 那条结论要改

| 轴 | 7.86 写的 | ★ 更正后 |
|---|---|---|
| **速度** | "同量级" | ★ **GT 1030 约为我们 CPU 的 2.7~3.0×** |
| 内存 | 我们 9× | ✓ 不变 |
| 精度 | 同类（都只有 FP32 算力） | ✓ 不变 |
| 功耗 | 15 W 对 30 W | ✓ 不变 |

**⇒ 而"没显卡"的代价因此【是真的】，不是被高估的：**

```
✗ 7.86 的乐观版: 峰值差 2.47×,而真实效率把差距拉回同量级
✓ 更正:         ★【峰值差 2.47×,而实际差距 2.7~3.0× —— 效率没有拉回,反而略放大】
                  （因为我们 CPU 在 2048³ 上只到 50%,而成熟库在 GPU 上到 56%,
                    再加上 GT 1030 的专用 48 GB/s 显存对我们共享的 ~25 GB/s）
```

#### 7.87.4 ★★ 而这带出了整件事真正的形状

**⇒ 我们【有一张 GT 1030 级别的 GPU】，而它被【功耗仲裁】废掉了：**

| 状态 | 核显 | 对 CPU 的 229.8 |
|---|---|---|
| **CPU 完全空闲** | ★ **644.9 GFLOPS** | **2.8×** |
| CPU 1 线程忙 | ~129 GFLOPS（÷5，7.62 实测） | **0.56×** |
| CPU 6 线程忙 | ~50 GFLOPS（÷13） | **0.22×** |

**⇒ ★【它一碰 CPU 就比 CPU 还慢 —— 而训练【不可能】让 CPU 空闲。**

**⇒ 所以"可恶的核显"这句话是【精确的】,而不是情绪：**

```
✗ "没显卡"       ⇒ 不成立 —— 我们有,而且它相当于一张 GT 1030
✗ "核显没用"     ⇒ 不成立 —— 它单独跑能到 644.9,是 CPU 的 2.8 倍
✓ ★【"核显的算力被共享的 15W 预算锁死了,而训练的 CPU 部分【必须】一直忙"】
⇒ ★【所以损失不是"没有 GPU",而是【有一个用不了的 GPU】】
```

#### 7.87.5 ★ 而这条与 §7.19.5 当时的推论形成完整闭环

**§7.19.5 当时的推理是：**

```
"若一设备显著快于另一,最优分配就是【全部给快设备】"
⇒ 而核显 644.9 对 CPU 229.8 ⇒ 2.8× ⇒ 当时的结论是【把 GEMM 交给核显】
```

**⇒ 而 §7.62 证明了那个前提不成立 —— 因为"给核显"意味着 CPU 空闲，
而训练里 CPU 有 47.6% 的活（attention + elementwise + autograd）【必须】做。**

**⇒ ★【所以 §7.19 的推理没错，是它的前提（两设备可独立满速）在这台机器上不成立。】**

**⇒ 而这个前提不成立的原因不在软件里 —— 在【15W 那个封装】里。**

#### 7.87.6 所以"没显卡"这件事的最终账（三改之后）

| 版本 | 速度结论 | 问题 |
|---|---|---|
| 7.86 | "同量级" | ✗ **把有锚点的量判成无法测,然后朝乐观方向取值** |
| **7.87** | ★ **GT 1030 约 2.7~3.0× 我们** | ✓ **锚点是我们自己测过的核显（同 384 shader）** |

**⇒ 而补救这件事的路径因此收窄成一条：**

```
✗ 让核显参与训练        ⇒ 功耗仲裁否决（7.62）
✗ 时间切片              ⇒ 1.62× 硬上限,代价是重写整套前反向（7.63）
✗ 指令层                ⇒ 无 SMT,且 GEMM 已到 50% 峰值（7.86 更正后）
✓ ★【库侧 —— 因为那是唯一不在这台机器 15W 预算之内的东西】
```

**⇒ 而 §7.86 那条关于内存的结论【仍然成立且更重要了】：**
**GT 1030 有 2 GB，我们有 15.4 GB ⇒ 100M 参数以上它【训不了】而我们可以。
⇒ ★【所以速度上我们输 2.7~3×,而规模上我们赢一整个档次。】**

'''

m = re.search(r'^### 7\.87 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.87'
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
