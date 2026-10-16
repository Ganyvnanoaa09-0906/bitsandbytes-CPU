"""Append 7.80: the 4-bit optimizer is 2.63x FASTER, not 4x slower. The docstring is inverted.

Chasing the three contradictions in bench_opt_4bit.py resolved all three, and the answer changes what
the 4-bit optimizer is. It was documented as a memory tool that costs speed; it is faster on both axes.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.80 ★ 4-bit 优化器【快 2.63×】，而不是"慢 4×" —— 头注释是反的

#### 7.80.1 起因：原基准里有三个矛盾

`bench_opt_4bit.py` 报出 4-bit 比 8-bit 快（比值中位 2.418），**却判"不可判定"**，
**而它的搬运量表印出两个优化器都是 1.031 字节/参数** —— 与 7.76 的 A/B（2.17 对 1.095）打架。

#### 7.80.2 三个矛盾的定位

**① 判据的 bug（`bench_opt_4bit.py:115`）**

```python
if abs(med - 1) > 2 * st.pstdev(ratios):      # ★ 错
```

**⇒ 它拿"中位数离 1 的距离"和"各个比值的离散度"比。而正确的尺度是【比值的标准误】
`pstdev/√n` —— 用原始离散度会把阈值放大 √n 倍（本次 n=9 时约 3 倍，n=25 时 5 倍）
⇒ 一个 2.4× 的效应被拿去和一个 3.6 的阈值比 ⇒ 判成噪声。**

**② 搬运量表的 bug（`:123-124`）**

```python
('8-bit (m+v+scale)', N, 2 * nb * 4)          # ★ 只算了【一个】状态
```

**⇒ 8-bit AdamW 的 m 和 v 是两个各 N 字节的 uint8 数组，而它只写了 N
⇒ 于是两行碰巧都算出 1.031 ⇒ 表看起来说"4-bit 一点没省"。**

**③ 没有离群值处理** —— 8-bit 那一列的 CV 是 55%，由一轮 5.063 ms（典型 1.5）造成，
**而 9 轮里一个这样的样本就能移动中位数。**

#### 7.80.3 修好之后的测量

| 统计量 | 8-bit ms | 4-bit ms | **8bit/4bit** |
|---|---|---|---|
| 中位 | 1.485 | 0.513 | **2.894×** |
| 均值 | 1.587 | 0.635 | **2.500×** |
| 截尾均值 10% | 1.541 | 0.627 | **2.459×** |

```
离散度: 8-bit CV 27.9% · 4-bit CV 33.9%（25 轮交替，每轮 10 次）
配对比值中位 2.627 · 标准误 0.253 · ★【95% CI 2.131 ~ 3.123】

★ 原判据: abs(med-1) > 2·pstdev ⇒ 阈值 2.53 ⇒ 报【不显著】（错）
★ 正确判据: 95% CI 是否排除 1 ⇒ ★【排除 ⇒ 显著】✓
```

#### 7.80.4 ★ 而结论与头注释【方向相反】

```
头注释: "the kernel currently has no AVX2 path, so it runs
         【about 4x slower】 than the 8-bit AVX2 kernel"
实测:   ★【4-bit 快 2.63×（CI 2.13~3.12）】
⇒ ★【方向相反,幅度差约 11 倍】
```

#### 7.80.5 而搬运量表修正后

| 项 | 字节/参数 |
|---|---|
| 8-bit m（1 个状态） | 1.000 |
| **8-bit m+v（★ 正确）** | **2.000** |
| **8-bit m+v+scale（每 64）** | ★ **2.125** |
| 4-bit m+v（打包） | 1.000 |
| **4-bit m+v+scale（每 256）** | **1.031** |
| fp32 Adam 参照 | 8.000 |

```
⇒ 8-bit 2.125 对 4-bit 1.031 ⇒ ★【省 2.06×】
⇒ 而原基准印 1.031 对 1.031 = "省 1.00×" —— 那是【8-bit 少算一个状态】的产物
⇒ ★ 而它与 7.76 的 A/B（8bit 2.17 · 4bit 1.095）吻合 ✓
```

#### 7.80.6 ★ 所以"它换的是什么"整个要改

| | 文档说的 | 实测 |
|---|---|---|
| 速度 | **慢约 4×** | ★ **快 2.63×** |
| 内存 | 省 7.8×（对 fp32） | 省 2.06×（对 8-bit，而那是对手） |
| 定位 | "能忍受就试试" | ★ **两边都赢** |

**⇒ 而那句 "about 4x slower" 【一直在让这个项目吃亏】—— 谁读了它都不会用这个优化器。**

#### 7.80.7 而这是今天第四条"资产被错误描述"

| # | 资产 | 错误描述 | 实际 |
|---|---|---|---|
| ① | 循环 Transformer | 以为没接 | 接了，`random=False` |
| ② | MaskGIT | 以为没有 | 完整实现，从未执行 |
| ③ | AVX2 4-bit 融合路径 | 以为缺实例 | 早已建好并调好阈值 |
| **④** | **4-bit 优化器** | **"慢 4×"** | ★ **快 2.63×** |

**⇒ 而第四条最严重：前三条是"不知道它存在"，而这条是【知道它存在、而被告知它不好】**
**⇒ 前者只是浪费机会，后者会【主动劝退使用者】。**

#### 7.80.8 而三条修正的优先级

```
① 改 adamw4bit.py 的头注释 —— 那句"慢 4×"必须删掉（它现在是错的,且方向相反）
② 改 bench_opt_4bit.py 的判据（L115）—— 用标准误,不用原始离散度
③ 改搬运量表（L123-124）—— 8-bit 必须算两个状态
④ ★ 而 7.77 那份"写 AVX2 路径"的规格【要重新评估】:
   既然它已经比 8-bit 的 AVX2 版快 2.63×,那"没有 AVX2 路径"这个前提本身就要查
```

**⇒ ④ 最关键：`adamw4bit.py` 说它"没有 AVX2 路径"，而实测它比【有】AVX2 路径的 8-bit 版还快
⇒ 所以要么那个内核其实有向量化，要么 8-bit 那条路才是慢的那个。
⇒ ★【而这正是 7.74 那条教训的第三次出现:一个"缺什么"的判断,必须先查再下。】**

'''

m = re.search(r'^### 7\.80 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.80'
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
