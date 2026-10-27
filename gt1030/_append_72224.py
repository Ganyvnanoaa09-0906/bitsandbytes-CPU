# -*- coding: utf-8 -*-
"""追加 §7.222.24：run C 结果 + 配对比较方法学（把 val 的误差从 ±0.052 收到 ±0.002）。"""
import io

TEXT = r"""

#### 7.222.24 run C 结果 + 一条测量方法学的修正：**配对比较**（±0.052 → ±0.002）

**run C（10-26 18:16 → 20:06）**：从 `ckpt_25000` 续、lr 5e-6→1e-6、带 NaN/spike 守卫与 opt-state。
它在 step 5084 被监督者按 plateau 规则停掉（理由是连续两个 ckpt 的 val 都没破 5.1963）：

| ckpt | val（382 张） |
|---|---|
| ckpt_27500（step 2500） | 5.2129 |
| ckpt_30000（step 5000） | 5.2081 |

**然后发现之前的"误差棒"用错了。** 单模型 val 的**非配对** SEM 是 **±0.052**（batch 间 sd 0.722 / √191），
拿它当尺子，0.01~0.05 的差都"不显著"，于是停止规则只能拍个 0.010 的容差——既不严谨也不灵敏。
正确做法是**配对比较**：同一批 val 图、同一 batch 划分，逐 batch 求两个模型的差：

```
python gt1030/ar_v2_valdump.py --ckpt <path> --out <dir>/x.npz   # 每个 ckpt 导一份逐 batch loss
python gt1030/ar_val_compare.py --dumps a.npz b.npz --base <name>
```

实测结果（n=191 个 batch，base = CPU 20k）：

| 模型 | val | Δmean（配对） | SEM(paired) | t |
|---|---|---|---|---|
| `ar_run_full3/ckpt_20000`（CPU 20k 基线） | 5.3551 | — | — | — |
| `ar_run_igpu40k/ckpt_25000`（runA +5k） | 5.2308 | **−0.1243** | 0.0020 | −62.1 |
| **`ar_run_igpu40k/ckpt_30000`（runA +10k，最佳）** | **5.1963** | **−0.1588** | 0.0027 | −59.1 |
| `ar_run_igpu_c/ckpt_30000`（runC +5k） | 5.2081 | −0.1470 | 0.0023 | −65.2 |

直接配对 runC vs runA 的 30k：**+0.0118 ± 0.0010（t = +12.4）⇒ run C 确实把它训坏了**（不是噪声），
所以监督者按 plateau 停止是**正确决策**；而 runA 的 25k 相对 30k 是 +0.0345 ± 0.0014（t=24.5）也显著更差。

**⇒ 两个结论**
1. **核显预训练的收益是统计显著的**：−0.1588 ± 0.0027（相对 CPU 20k），不是"看起来降了 0.16"。
2. **配对 SEM 比非配对小 20~50 倍**（0.0010~0.0027 vs 0.052）。以后**所有模型选择都必须用配对比较**；
   非配对 SEM 只适合回答"这个模型的绝对水平大概多少"，不适合比较两个模型。
   工具已留在 `gt1030/ar_v2_valdump.py` + `gt1030/ar_val_compare.py`。

**当前最优模型**：`F:\weibo_cos\ar_run_igpu40k\ckpt_30000.pt`（val 5.1963，配对意义下优于所有其它点）。
`ar_run_igpu_c\ckpt_30000.pt`（5.2081）和 `ckpt_27500.pt`（5.2129）保留作对照；两者都有 `.opt.npz`，
可以作为"热状态续跑"的起点再做实验。

**"继续训还有没有用"的判据**（用配对比较，不用看绝对值）：新 ckpt 相对 5.1963 的 Δ 若 **+0.005 以上且 t>2**
就是真退步；若落在 ±0.003 内则是**平台**（再多训也不涨）；只有 Δ < −0.005 才值得继续。
按这个尺子，run C 的两次尝试（+0.0166、+0.0118）都落在"真退步"，说明 **lr 5e-6 从 25k 起跑只会轻微破坏模型**，
不是"训练量不够"，而是这个配方在这个点上已经没有上升空间。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s += """
## run C 结果 + 配对比较（10-27 早）

- run C 在 step 5084 被 plateau 规则停掉：ckpt_27500 = 5.2129、ckpt_30000 = 5.2081，都没破 5.1963
- **配对比较（同 val 集逐 batch 差）**：SEM 从 ±0.052 收到 **±0.001~0.003**
  - runA 25k: −0.1243 ± 0.0020 (t=−62)
  - **runA 30k: −0.1588 ± 0.0027 (t=−59) ← 最佳**
  - runC 30k: −0.1470 ± 0.0023，直接配对 runC−runA30k = **+0.0118 ± 0.0010 (t=+12.4) 真退步**
- 结论：核显 10k 步的收益统计显著；lr 5e-6 从 25k 续跑只会轻微破坏模型（平台，不是训练量不足）
- 工具：`gt1030/ar_v2_valdump.py` + `ar_val_compare.py`（以后模型选择都用配对）
"""
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')
