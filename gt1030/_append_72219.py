# -*- coding: utf-8 -*-
"""追加 §7.222.19：优化后整步 1.208 s（干净测量），以及"软 max 已到带宽瓶颈"的负结果。"""
import io

TEXT = r"""

#### 7.222.19 优化后整步实测 **1.208 s**（1030 分数 0.43 → 0.49），以及 softmax 的负结果

**干净测量**（直接测 `fwd_body`/`bwd_body`/整步，各取 4~6 次平均，机器相对安静时）：

| 项 | 优化前 | 现在 | 变化 |
|---|---|---|---|
| `Block.fwd_body`（单层，不含传输） | 46.7 ms | **36.1 ms** | −10.6 ms/层 |
| `Block.bwd_body`（单层） | 91.0 ms | **67.1 ms** | −23.9 ms/层 |
| **整步** | 1.383 s | **1.208 s** | **−175 ms（−12.7%）** |
| 其中 fwd / bwd / opt | 0.434 / 0.799 / 0.102 | 0.394 / 0.722 / 0.088 | — |

⇒ 1030 分数 = 0.59 / 1.208 = **0.49**（起点 0.21，移植前是 0.43）。

**负结果 1：向量化 softmax 没有收益。** 把两个 causal softmax 改成 half8 向量读写后：
`softmax_causal` 3.32 → 3.36 ms、`softmax_bwd_causal` 3.39 → 3.34 ms（**0.99×/1.01×**）。
原因是它们已经是**带宽瓶颈**：每次读 32 MB + 写 32 MB，3.3 ms 对应约 **19 GB/s**，
已经接近这台机共享 DDR4 的可用带宽。⇒ **softmax 想快，唯一的办法是别把 S/P 落盘**（真 flash），
向量化在这里没有意义。这也修正了 §7.222.16 里"attention 融合能省 200 ms"的估计：
按现在的实测，flash 化实际能省的是 **S/P/dP/dS 的落盘流量约 100 ms/步**，不是 200。

**当前时间构成（1.208 s）**：

| 块 | 耗时 | 占比 | 状态 |
|---|---|---|---|
| 大 GEMM（前 4 + 后 7，共 11 个/层） | ~560 ms | 46% | CLBlast 497~700 GFLOPS ≈ 本机 fp16 实际可达，**但没做双缓冲** |
| attention（S/dP/O/dQ/dV/dK + softmax） | ~260 ms | 22% | 已用自写 kernel + 因果裁剪；再快要靠真 flash |
| 小 kernel + AdamW + head + 传输 | ~390 ms | 32% | 已向量化/融合过一轮；opt_step 88 ms、grad_norm 12.6 ms |

**下一步判断**：GEMM 那 560 ms 更值得动 —— 现在每个 k-迭代是"载入 8KB → 算 → 屏障"，
**载入和计算没有重叠**，这正是只跑到 50% 效率的原因；用 `async_work_group_copy` 做双缓冲是标准解法，
预期能到 1.3~1.5×（省 150~200 ms），收益比 flash 化更大且风险更低。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))
