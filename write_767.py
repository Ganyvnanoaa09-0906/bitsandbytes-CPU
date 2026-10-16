"""Append 7.67: the three headaches, and what is actually deliverable.

The user framed the project's pressure as three things: no GPU, video training, and slow pretraining.
Those have now each been measured to a conclusion, and only one of them yields something usable. This
section puts the three answers side by side, because the pattern across them is the same and is worth
naming.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.67 三个头疼的账：只有一条给出了可交付的东西

用户把项目的压力归结为三件事：**没显卡 · 视频训练 · 龟速预训练**。
而它们各自测到了一个结论，而三条的**形状是同一个**。

#### 7.67.1 没显卡 —— 三条路全关

| 路线 | 理论 | 实测 | 状态 |
|---|---|---|---|
| 指令层面（填气泡 / 一条指令干两件事） | ≤2.5× | — | ✗ **无 SMT，结构性死角** |
| 工程（核显卸载） | 2.05× | **负**（功耗仲裁，开关式） | ✗ |
| 工程（时间切片） | 1.62~1.96× | **1.24~1.62×（硬上限）** | ✗ 代价是重写整套前反向 |
| 数学（Monarch） | 10~28× FLOPs | **0.81~1.30× 壁钟** | ✗ FLOPs 不翻译成壁钟 |

**⇒ 结论：这台硬件上，训练侧没有可用的加速。**

#### 7.67.2 ★ 视频训练 —— VCN 硬解可用，而这是唯一可交付的

```
解码方式       帧数    墙钟 ms    CPU ms    CPU/帧 ms
软解           120     43.5     109.4     0.9115
★ d3d11va     120     43.9     【46.9】   0.3906   ← CPU 省 2.33×
dxva2         120     45.0      78.1     0.6510
auto          120     43.6     156.2     1.3021   ← ✗ 静默变慢 1.43×,别用
```

**⇒ 而它在真实训练循环里的收益是【2.9%@640×360】，而它随分辨率增长：**

| 分辨率 | 解码占步时 | 硬解的价值 |
|---|---|---|
| 640×360 | 12% | **2.9%（判据未过，但免费）** |
| 1280×720 | 约 48% | **必须用** |
| 1920×1080 | 约 108% | ★ **解码比模型还贵** |

**⇒ 用法一行：`av.open(path, options={'hwaccel': 'd3d11va'})`**

**⇒ 而它为什么能用，与着色器核心为什么不能用，是同一件事的两面：**

```
着色器核心:和 CPU 抢同一个 15W ⇒ 功耗仲裁 ⇒ 一碰掉 5 倍
★ VCN:固定功能 ⇒ 不占 FP 单元、不跑着色器 ⇒ 【它不是抢饭,是多了一个工人】
```

#### 7.67.3 龟速预训练 —— 训练侧已到下限，而杠杆在库侧

```
epoch 算术:0.43 个 epoch(实测),一个 epoch 17 小时
四个速度杠杆全关:GEMM 79~85% 形状最优 · attention 已最优 · elementwise 已融合 · 核显否掉
⇒ ★【唯一值一个数量级的活,在【库】那一侧】
```

**⇒ 而那个活已经有一个实例，且已完成：**

```
gemv_4bit_inference 上游【只有 bf16 实例】⇒ AVX2-only 机器静默回退
⇒ 回退路径 = 反量化整个权重 + 稠密 GEMM
⇒ 实测:R5-4500U 上 12.3 ms 对 1.16 ms ⇒ ★【10.6×】
```

**⇒ 而按同一形状清点，还有两处未查：`tinygemm_kernel_nn`（整块被 `#if` 围住）、
以及整个后端零个整数点积指令（2~4.5×）。**

#### 7.67.4 ★ 而三条的形状是同一个

```
没显卡     ⇒ 想在【硬件】上要速度 ⇒ 开关式约束(功耗仲裁)⇒ 无解
视频训练   ⇒ 换一个【部件】(VCN,固定功能)⇒ 规避了那个约束 ⇒ 有解 ✓
预训练慢   ⇒ 换一个【层面】(库,而非训练)⇒ 绕开了那个约束 ⇒ 10.6× ✓
```

**⇒ 所以"没显卡"这件事的正确回应不是"再榨这台机器"，而是：**

> **当一个约束是【开关式】的（要么全有要么全无），不要在它内部找中间值 ——
> 去换一个【不受那个约束管辖】的部件或层面。**

* 功耗仲裁管的是**着色器核心** ⇒ 换 VCN（固定功能，不在管辖内）
* 训练步的下限管的是**这台机器的训练** ⇒ 换库（那是另一条产品线）

#### 7.67.5 而这也解释了本项目最有价值的产出是什么

**它不是某个模型，也不是某个加速比，而是【一张"哪些约束是开关式的、以及绕过它们的部件在哪"的地图】：**

| 开关式约束 | 管辖范围 | 绕过它的东西 |
|---|---|---|
| **功耗仲裁**（CPU 与核显满速互斥） | 共享 15W 封装里的**着色器核心** | **VCN**（固定功能） |
| **无 SMT**（发射气泡填不满） | **FP32 FMA 吞吐** | 整数指令（`VPMADDUBSW`） |
| **静默回退**（模板只覆盖新硬件） | **AVX2-only 机器上的量化路径** | **融合内核（已做，10.6×）** |
| **算力上限**（600 GFLOPS） | **这台机器能训的模型规模** | 缩小任务（网格/数据量） |

**⇒ 四行，而每一行都是"一个开关 + 一个绕开它的东西"。**

**⇒ 而这就是"没显卡"这个问题在工程上唯一能被回答的方式。**

'''

m = re.search(r'^### 7\.67 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.67'
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
