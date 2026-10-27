# -*- coding: utf-8 -*-
"""追加 §7.222.13：向量化重写 attention GEMM + grad_norm 修复，一步 2.075 → 1.616s（比 CPU 快 2.23×）。"""
import io

TEXT = r"""

#### 7.222.13 向量化 GEMM 重写：一步 **2.075 → 1.616 s**（972 → 1267 tok/s，比 CPU 快 2.23×）

**先纠正一件事**：§7.222.6 说"自写朴素 tiled GEMM 打不过 CLBlast"——那是在 CPU 满载污染的
条件下、且用的是**标量** local 读。改成下面两件事后，自写 kernel 全面超过 CLBlast：

1. **local 里存转置后的 tile，让计算循环能用 `vload4` 向量读**
   A 在 global→local 时转成 `At[k][m]`，于是每次 kk 只需 2 条 LDS 指令（half4）换来 16 次 FMA；
   原来是 8 条标量 LDS。GCN 上 LDS 指令吞吐就是瓶颈（每 warp 一条要 ~4 拍）。
2. **按形状特化**：attention 的 batched GEMM 分两类，
   - `S = Qn·Knᵀ`、`dP = dO·Vbᵀ`：k=64 可以整个装进 local（At/Bt 各 8~16KB），128×64 tile、每线程 8×4
   - `O = P·Vb`、`dQ = dS·Kn`：k=T=1024 需要切块（BK=64），A 转置进 local、B 直接落位

**独占口径实测（CPU trainer 挂起）**：

| 算子 | CLBlast | 自写 | 加速 |
|---|---|---|---|
| S = Qn·Knᵀ（16×1024×1024×64） | 14.98 ms / 143 GFLOPS | **8.40 ms / 256 GFLOPS** | **1.78×** |
| dP = dO·Vbᵀ（同形状） | 15.19 ms | ~8.4 ms | 1.8× |
| O = P·Vb（16×1024×64×1024） | 9.82 ms / 219 GFLOPS | **6.54 ms / 328 GFLOPS** | **1.50×** |
| dQ = dS·Kn（同形状） | 9.86 ms | 6.54 ms | 1.5× |

数值都对：k=64 kernel rel 2.9e-3、nn kernel rel 1.2e-2（都是 fp16 累加的底线），
整机 10 步轨迹复测 **最大偏差 1.99e-3**（与优化前 2.00e-3 一致，说明没改坏语义）。

**另一个大坑：`grad_norm` 花了 200 ms。** 它把 40 个权重梯度（**59 MB**）从 device 拷回 host
再算平方和 —— 每步 200ms，占总步时的 10%。改成设备端 `sumsq_half` 归约（每个参数只回传 256 个
部分和 = 1KB），并且把 41 次调用**批量 enqueue、只同步一次**（原来每次 `sumsq` 都 `finish()`，
光驱动同步就 ~80ms）：**200ms → 18ms**。

**累计效果（都是独占口径）**：

| 版本 | 一步 | tok/s | 相对 CPU(569) |
|---|---|---|---|
| 原始（且被 CPU 训练污染） | 4.05 s | 505 | 0.89× |
| 独占、未优化 | 2.075 s | 987 | 1.73× |
| + 自写 S/dP k=64 GEMM（1.78×） | 1.833 s | 1117 | 1.96× |
| + 自写 O/dQ batched nn GEMM（1.50×） | 1.777 s | 1152 | 2.02× |
| + grad_norm 设备端归约（200→18ms） | **1.616 s** | **1267** | **2.23×** |

**剩下的时间在哪（独占剖面）**：前向 0.50s、反向 1.00s、优化器 0.10s。
反向里大 GEMM（7 个）约 500ms、attention 约 250ms、其余小 kernel（rms/split/qk-norm/silu/add）约 250ms。
大 GEMM 已经跑在 CLBlast 的 500~675 GFLOPS（≈ fp16 峰值的 45~59%），**这块基本没肉了**；
还能动的是：attention 的 softmax 与 S 融合（省 6.7ms/层）、以及 20 多个小 kernel 的合并（约 400ms/步的盘子）。

**产物**：`gt1030/gemm_half_attn.py`（k=64 batched）、`gt1030/gemm_half_nn_batched.py`（k=T batched）、
`gt1030/gemm_half_v2.py`（通用 nn 扫描）、`gt1030/_t_fair_bench.py`/`_t_fair_prof.py`/`_t_fair_train.py`
（挂起 CPU 的公平测量工具）。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace("**吞吐结论（更正版）**：核显独占 **2.11 s/step = 972 tok/s**；CPU 569 tok/s；两者并发时核显掉到 4.05s/step。",
              "**吞吐结论（更正+优化后）**：核显独占 **1.616 s/step = 1267 tok/s**（比 CPU 的 569 快 **2.23×**）；"
              "优化路径：2.075（独占基线）→1.833（自写 k=64 GEMM）→1.777（自写 O/dQ GEMM）→1.616（grad_norm 设备端归约）。"
              "两者并发时核显掉到 4.05s/step。")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')
