# -*- coding: utf-8 -*-
"""追加 §7.222.6：自写 attention GEMM 的实测（负结果，但把数据留下）。"""
import io

TEXT = r"""

#### 7.222.6 自写 attention GEMM 的实测：能算对，但比 CLBlast 还慢

结论先放：**"换个自己写的 tiled GEMM 就能提速"这个想法，在没做向量化+寄存器分块之前是错的。**

`S = Qn·Knᵀ`（16 个 batch、(1024×1024×64)）是一个 k 很薄的 batched GEMM，CLBlast 只跑到
63.5 GFLOPS，所以先试了自写：64×64 tile、`__local` 装 Q/K 两个 tile（各 8KB）、256 线程、
每线程 4×4 输出、fp32 累加、1D 工作组（`group_id` 分解出 bh/ti/tj）。

| 实现 | 时间 | GFLOPS | 相对 |
|---|---|---|---|
| CLBlast `HgemmStridedBatched` | 33.8 ms | 63.5 | 1.00× |
| 自写 64×64 tiled（朴素标量） | 55.9 ms | 38.4 | **0.60×** |
| 自写 + 16 次单 GEMM（带偏移）代替 batched | 95.0 ms | 22.6 | **0.40×** |

数值是对的（vs numpy rel 3.2e-4；CLBlast 那条 7.0 的"误差"是因为 scale 本来就在 softmax kernel 里乘）。

三条可以写进经验：
1. **不要把 batched 拆成 N 次单 GEMM** —— 在这台机器上反而慢 2.5×（每次调用的固定开销太大）。
2. **朴素 tiled kernel 打不过 CLBlast**：它内部做了向量化（half2/half4）和寄存器分块；报告 §A.14 里
   自写 half GEMM v1/v2 也只有 352~416 GFLOPS，同样不如 CLBlast 的 1039。要赢它得做真正的
   register-blocked + 向量化 + double buffering 的 kernel，那是「几天」级别的工作，不是顺手能做。
3. 因此 attention 的优化收益必须主要来自**算法层**：causal-tiled（只算下三角，省一半 FLOPs）
   和 fused softmax（不物化 S/P/dP/dS，每层省 4×32MB 读写），而不是换 GEMM 实现本身。

**当前的性价比判断**：M4/M5/M6 的链路已经打通且数值正确，但要把核显训练做到比 CPU（3.1s/step）快，
需要「向量化 fp16 GEMM + flash attention」这一整套，属于后续独立工程。**在那之前，生产训练仍留在 CPU；
核显这条线的价值是：算法侧实验（改结构/改 loss 的快速迭代）和推理侧（§7.222 的目标本来就是"图像与视频"）。**
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))
