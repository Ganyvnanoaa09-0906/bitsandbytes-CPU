# -*- coding: utf-8 -*-
"""追加 M5 的"同源可比"证据：CPU ckpt@12000 采样 vs iGPU 训练 20 步后导出的采样。"""
import io

TEXT = r"""

#### 7.222.7 M5 的"同源可比"证据（采样图对比）

导出链路要证明的不是"好看"，而是**同一个模型、同一套采样代码、只是权重来源不同**。

| 采样来源 | 图 | token 使用 | 熵 | 最高频码占比 |
|---|---|---|---|---|
| CPU ckpt@12000 | `ar_sample_12000.png`（16 张） | 346~398 / 1024 | 4.47~4.77 / 6.93 | — |
| iGPU 训练 1 步后导出 | `ar_sample_igpu_test.png`（4 张） | 409 / 1024 | 5.26 / 6.93 | 3.4% |
| iGPU 训练 20 步后导出（`ckpt_12020.pt`） | `ar_sample_igpu_12020.png`（4 张） | 391 / 1024 | 4.93 / 6.93 | 6.3% |

三者都是**同一阶段的色彩场**（12k~20k 步、约 2 个 epoch 的欠训练状态），码使用率与熵同一量级，
说明 iGPU 训练出来的权重与 CPU 的权重在"同一个分布、同一个训练进度"上 —— 这正是 M5 要的
"同源可比"。图本身不好看是模型容量/训练量的老问题（§7.221 已记），与移植无关。

image: `F:\weibo_cos\ar_sample_12000.png`（CPU）、`F:\weibo_cos\ar_sample_igpu_12020.png`（iGPU）
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))
