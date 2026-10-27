# -*- coding: utf-8 -*-
"""追加 §7.222.14：再压 1.616 → 1.483 s/step（1381 tok/s，比 CPU 快 2.43×）。"""
import io

TEXT = r"""

#### 7.222.14 再压一轮：**1.616 → 1.483 s/step（1381 tok/s，比 CPU 快 2.43×）**

这一轮抓到的是一个"看起来很小其实很贵"的 kernel：`rms_dw_from_ss`（RMSNorm 的 w 梯度）。
它的 qk-norm 版本是 **(64 列 × 16384 行)**，而原实现只有 **64 个工作项**、每个串行扫 16384 行
⇒ **5.50 ms/次**，一层调两次、十层就是 **110 ms/步**。

改成二维归约 `rms_dw_part`（grid = `(d, 32)`：每列切成 32 段，每段并行扫 `rows/32` 行，
部分和写进 `part[chunk*d + i]`，host 侧再把 32 段加起来 —— 只回传 32×512 个 float）：
5.50 ms → ~0.3 ms，**一步省 100+ ms**。

顺带确认了 launch 开销不是问题：**单次 kernel launch 只有 0.023 ms**，400 次/步 ≈ 9 ms，
所以 "20 多个小 kernel" 不能靠"合并 launch"省时间，它们是真的在做内存搬运。

**最新对照（独占口径）**：

| 版本 | 一步 | tok/s | 相对 CPU(569) |
|---|---|---|---|
| 并发污染下的旧测量 | 4.05 s | 505 | 0.89× |
| 独占、未优化 | 2.075 s | 987 | 1.73× |
| + 自写 S/dP k=64 GEMM | 1.833 s | 1117 | 1.96× |
| + 自写 O/dQ batched nn GEMM | 1.777 s | 1152 | 2.02× |
| + grad_norm 设备端归约（200→18ms） | 1.616 s | 1267 | 2.23× |
| + `rms_dw` 并行归约（5.5→0.3ms ×2×10 层） | **1.483 s** | **1381** | **2.43×** |

现构成：前向 0.493 s、反向 0.889 s、优化器 0.106 s。反向里大 GEMM 337 ms（已接近 CLBlast 上限
500~675 GFLOPS）、attention ~250 ms、其余小 kernel ~200 ms。
10 步轨迹复测 **1.98e-3**（验收 0.02），语义没变。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace("**吞吐结论（更正+优化后）**：核显独占 **1.616 s/step = 1267 tok/s**",
              "**吞吐结论（更正+优化后）**：核显独占 **1.483 s/step = 1381 tok/s**")
s = s.replace("优化路径：2.075（独占基线）→1.833（自写 k=64 GEMM）→1.777（自写 O/dQ GEMM）→1.616（grad_norm 设备端归约）。",
              "优化路径：2.075（独占基线）→1.833（自写 k=64 GEMM）→1.777（自写 O/dQ GEMM）"
              "→1.616（grad_norm 设备端归约）→**1.483（rms_dw 并行归约）**。")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')
