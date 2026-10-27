# -*- coding: utf-8 -*-
"""追加 §7.222.15：利用因果结构跳过上三角 → 1.338 s/step（1531 tok/s，比 CPU 快 2.69×）。"""
import io

TEXT = r"""

#### 7.222.15 利用因果结构：**1.483 → 1.338 s/step（1531 tok/s，比 CPU 快 2.69×）**

这一轮不需要新 kernel，只是把"causal"这件事用起来。关键前提（先查了 kernel 源码确认）：
`softmax_causal` 只读 `k < row+1`、并把 `k ≥ row+1` 显式写成 0；`softmax_bwd_causal` 同理。
所以 attention 链路里**严格上三角的那部分从来不被读**，可以整个不算：

| 算子 | 怎么做 | 省 |
|---|---|---|
| `S = Qn·Knᵀ`、`dP = dO·Vbᵀ` | tile (ti,tj) 当 `tj` 完全在 `ti` 右侧时直接 `return` | ~50% |
| `O = P·Vb`、`dQ = dS·Kn` | A 的第 r 行块只需要 `k ≤ r` 的列块（P/dS 上三角恒为 0）⇒ k 循环只跑到 `mb+128` | 平均 ~50% |

被跳过的项**数学上恒等于 0**，所以结果完全不变——实测 loss 前后都是 4.8599（逐位相同），
10 步轨迹复测 **1.98e-3**。

**完整优化链（每行都是独占口径、同一台机器）**：

| 版本 | 一步 | tok/s | 相对 CPU(569) |
|---|---|---|---|
| 并发污染下的旧测量（错误结论） | 4.05 s | 505 | 0.89× |
| 独占、未优化 | 2.075 s | 987 | 1.73× |
| + 自写 `S/dP` k=64 GEMM（1.78×） | 1.833 s | 1117 | 1.96× |
| + 自写 `O/dQ` batched nn GEMM（1.50×） | 1.777 s | 1152 | 2.02× |
| + `grad_norm` 设备端归约（200→18 ms） | 1.616 s | 1267 | 2.23× |
| + `rms_dw` 并行归约（5.5→0.3 ms） | 1.483 s | 1381 | 2.43× |
| + 因果结构跳上三角 | **1.338 s** | **1531** | **2.69×** |

**现在的构成**：前向 0.434 s、反向 0.799 s、优化器 0.102 s。
反向里大 GEMM 337 ms（CLBlast 500~675 GFLOPS，已接近本机 fp16 实际上限）、attention 约 180 ms、
其余小 kernel 约 280 ms。**从 2.075 到 1.338 是 1.55× 的纯工程收益，且全程语义不变（轨迹 1.98e-3）。**
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace("**吞吐结论（更正+优化后）**：核显独占 **1.483 s/step = 1381 tok/s**",
              "**吞吐结论（更正+优化后）**：核显独占 **1.338 s/step = 1531 tok/s**")
s = s.replace("→**1.483（rms_dw 并行归约）**。", "→1.483（rms_dw 并行归约）→**1.338（因果跳上三角）**。")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')
