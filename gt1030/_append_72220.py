# -*- coding: utf-8 -*-
"""追加 §7.222.20：收尾测量（1.216 s/步 = 1684 tok/s）+ 两个负结果 + 最终评估。"""
import io

TEXT = r"""

#### 7.222.20 收口：**1.216 s/步 = 1684 tok/s（2.96× CPU，1030 分数 0.485）** + 两个负结果

**干净窗口的最终测量**（60 步，机器安静、CLBlast 大 GEMM 回到 460~686 GFLOPS 的那个时段）：

| 项 | 值 |
|---|---|
| median | **1.216 s/step** |
| min / mean | 1.196 / 1.218 s |
| 吞吐 | **1684 tok/s**（每步 2048 token） |
| 对照 CPU（569 tok/s） | **2.96×** |
| 对照 GT 1030 roofline（0.59 s） | **1030 分数 0.485** |

优化链全程：CPU 2.854 s（0.21）→ 核显初版 2.075 s（0.28，但被并发污染测成 4.05 s）
→ 自写 attention GEMM 1.833 → O/dQ GEMM 1.777 → grad_norm 设备端归约 1.616
→ rms_dw 并行归约 1.483 → 因果跳上三角 1.338 → **向量化/融合 + 层间 dev→dev + opt_step 合并同步 1.216**。

**负结果 1：向量化 softmax 无效（0.99×/1.01×）。** 它们是带宽瓶颈（64 MB/3.3 ms ≈ 19 GB/s），
不是指令瓶颈。⇒ 也修正了 flash 化的收益估计：能省的是 **S/P/dP/dS 的落盘流量约 100 ms/步**，
不是之前估的 200 ms。

**负结果 2：GEMM 双缓冲这条路在 OpenCL 上被 API 卡住。**
现在每个 k 迭代是"载入 8KB → 屏障 → 计算 → 屏障"，载入期间 ALU 空转，这是只跑到 40~60% 效率的原因。
标准解法是 `async_work_group_strided_copy` 做 DMA 预取，**但它是 1-D 跨步复制**：
一次调用只能拷"num 个元素、源间隔 stride"，无法直拷"MS 行 × BK 列、行跨步 K"的 2-D tile
（我按这个思路写的版本 rel=1.07，数值错，因为 stride 语义是"逐元素跨步"而非"逐行跨步"）。
可行的替代要么每行发一次 DMA（128 次调用，开销吃掉收益），要么先直拷再在 local 内转置（多一倍 local 流量）。
⇒ **CLBlast 的 460~686 GFLOPS 基本就是这台 6 CU 核显在这批形状上的实际可达值**，
我们的自写 GEMM 与它 0.86~1.11×，双缓冲没有便宜的实现路径。

**最终评估（把 1030 这件事讲清楚）**：

| 口径 | 结果 |
|---|---|
| fp16 纯运算 / 大 GEMM | 核显 1290 / 1039 GFLOPS vs 1030 理论 1127、可达 ~631 ⇒ **1.14~1.65×，早就是"跑分赢"** |
| 本模型单步（375 GFLOP） | 核显 1.216 s vs 1030 roofline 0.59 s ⇒ **0.485**（从 CPU 的 0.21 提升 2.3×） |
| 显存 | 我们需要 2.85 GB 常驻，**1030 只有 2 GB —— 它跑不了这个 workload**（要梯度检查点+优化器 offload，速度再掉 30~50%） |

**结论**：速度上还没"干翻"1030 的 roofline（差 2.1 倍），差的这 2.1 倍里
**46% 是大 GEMM（已到 CLBlast 上限）、22% 是 attention（再快要真 flash，约 100 ms）、32% 是其余开销**。
而"能训 vs 训不了"这条口径上，我们是赢的（2.85 GB vs 2 GB）。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace('| 下一步 | attention 融合（flash 风格），目标 ~1.1 s/步 |',
              '| 最终 | **1.216 s/步 = 1684 tok/s = 2.96× CPU，1030 分数 0.485**（60 步、干净窗口） |\n'
              '| 负结果 | 向量化 softmax 无效（带宽瓶颈 19 GB/s）；GEMM 双缓冲被 `async_work_group_strided_copy` 的 1-D 语义卡住 |')
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')
