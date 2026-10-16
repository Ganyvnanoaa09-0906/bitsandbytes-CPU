"""Append 7.56: Monarch passes its self-check and fails its speed test. The 10x is unreachable.

The self-check finally passes after fixing an einsum contraction that computed x @ R where the
matrix form computed R @ x. With that corrected, all five of the model's shapes reproduce their own
effective weight matrix to 1e-6, and the parameter reduction is 10x to 28.5x.

Then the speed test, which is the one that matters. On this CPU the layered form is at best 1.30x
faster in wall clock and 0.81x on the square case, against a criterion of 2x fixed in advance.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.56 Monarch 自检通过、速度不通过 —— 那道 10 倍的坎跨不过去

#### 7.56.1 而上次的误诊要更正

**7.55 把 6.26 的误差归给"置换方向反了"，那是错的。** 逐中间量查出来是：

```
步骤1 (Rm): 误差 5.18  ← 就是这里
步骤2 (Q) : 误差 0.00  ← 置换其实是【对的】

einsum 'abi,bij->abj' 算的是  out[j] = Σᵢ x[i]·R[i,j] = x @ R
而矩阵版 Rm @ x 算的是                    R[i,j]·x[j] = R @ x
⇒ ★【两者互为转置】
```

**修法**：`'abi,bij->abj'` → **`'abi,bji->abj'`**。

**⇒ 教训**：**"误差很大"不等于"结构错"** —— 而我把一个【契约方向的错】诊断成了【结构错】，
两者修改的位置完全不同。**⇒ 定位要靠中间量，不靠猜。**

#### 7.56.2 自检：全部通过

| 形状 n→m | b_r/b_l | 参数 | 稠密 | 倍数 | 自检误差 |
|---|---|---|---|---|---|
| 512→512 | 16/16 | 16384 | 262144 | **16.0×** | 1.43e-06 ✓ |
| 512→2048 | 16/32 | 73728 | 1048576 | 14.2× | 1.07e-06 ✓ |
| 2048→512 | 32/16 | 73728 | 1048576 | 14.2× | 1.25e-06 ✓ |
| 2048→2736 | 32/48 | 196864 | 5603328 | **28.5×** | 1.43e-06 ✓ |
| 2736→512 | 48/16 | 139520 | 1400832 | 10.0× | 1.43e-06 ✓ |

**⇒ 判据（误差 < 1e-5）全部通过。**

#### 7.56.3 速度：不通过

**模型的真实批次（2048 行 = batch 2 × 1024 token）：**

| 批次×n→m | 稠密 ms | Monarch ms | **倍数** | FLOPs 比 |
|---|---|---|---|---|
| 2048×512→512 | 3.815 | 4.701 | **0.81×** ✗ | 16.0× |
| 2048×512→2048 | 21.420 | 16.420 | **1.30×** | 14.2× |
| 2048×2048→512 | 15.991 | 13.956 | 1.15× | 14.2× |
| 2048×512→2736 | 29.364 | 22.644 | **1.30×** | 10.0× |
| 2048×2736→512 | 21.785 | 17.231 | 1.26× | 10.0× |

**判据（事先固定：壁钟 ≥ 2×）⇒ 【不通过】✗**

#### 7.56.4 为什么 FLOPs 降 10~28× 而壁钟只快 1.15~1.30×

**块对角 einsum 跑的是一堆【小矩阵乘】而不是一个大矩阵乘，而小矩阵乘恰好是这台机器最弱的：**

* 本项目已记录：**NS 的批量版比逐矩阵版慢 11%**
* **核显在小矩阵上比 CPU 慢 1.55×**（7.28）
* 而 `xr[:, perm]` 还要一次 n 元素的 gather

**⇒ 数学上的 10 倍是真的，但它翻译不到壁钟上。**

#### 7.56.5 ★ 于是三条路的账最终是

| 路线 | 理论 | 实测 | 状态 |
|---|---|---|---|
| 指令层面（填气泡/一条干两件事） | ≤2.5× | — | ✗ 无 SMT，结构性死角 |
| **工程（核显卸载）** | — | **1.78×** | ✓ **可用，但由 CPU 那 47.6% 封顶** |
| **数学（Monarch）** | **10~28×** | **0.81~1.30×** | ✗ **FLOPs 不翻译成壁钟** |

**⇒ ★【那道 10 倍的坎跨不过去。】**

**⇒ 而这意味着 epoch 账的最终结论**：

```
最优可得 ≈ 1.78× ⇒ 一个 epoch 17 小时 ⇒ 【9.6 小时】
⇒ 10 个 epoch ⇒ 【约 4 天】
⇒ 而图像生成模型的通常做法是 100 个 epoch ⇒ 【40 天】
⇒ ★【这台机器无法在合理时间内训出这个任务的充分训练模型】
```

#### 7.56.6 所以真正剩下的三条出路

1. **缩小任务** —— 更少的图、更小的模型、更粗的网格（7.50 的 24× 组合）
   **⇒ 它不改硬件需求，改的是问题的规模 ⇒ 唯一能立刻执行的**
2. **换硬件** —— 而 1030 只值 2.5×，要的是 10× 以上 ⇒ **2060 级（带 tensor core）**
3. **接受边界** —— 把项目的产出定位为【方法】而不是【模型】
   （Muon 3.57× · swizzle 2× · 权重衰减修复 · 以及十四条方法论）

**⇒ 而第 1 条是今晚就能做的，而它需要的正是记忆测试的结论。**

'''

m = re.search(r'^### 7\.56 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.56'
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
