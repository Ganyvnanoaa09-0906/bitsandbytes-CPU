"""Append 7.55: Monarch step one -- the implementation fails its own check.

Kept short deliberately. The substance is also in the commit message and in monarch_layer.py's
docstring; this is the version that belongs in the document of record.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.55 Monarch 第一步：实现未通过自检，而这就是它该有的样子

#### 7.55.1 一个写在动手之前的设计决定

**诱人的做法**：拿训好的稠密权重，用 Monarch 去近似它，看误差随块大小 b 怎么变。
**⇒ 但那测的是压缩率，不是层** —— 任意矩阵要 b = n 才能精确表示。

**⇒ 所以改成**：**Monarch 层【替换**稠密层、从头训练**（那才是真正省算力的做法）。
**而那时"等价性"的含义变成：层算的是不是它自己的有效权重矩阵。**

**⇒ 判据（事先固定）：误差 < 1e-5 才允许开始计时。**

#### 7.55.2 自检结果：不通过

| 形状 n→m | b_r/b_l | 参数 | 稠密参数 | 倍数 | **误差** |
|---|---|---|---|---|---|
| 512→512 | 16/16 | 16384 | 262144 | **16.0×** | **6.26e+00** ✗ |
| 512→2048 | — | — | — | — | 有效权重形状 (512,512) ≠ (2048,512) |
| 2048→2736 | — | — | — | — | 同上 |

**⇒ 判据不通过 ⇒ 【不报任何性能数字】。**

#### 7.55.3 而这就是顺序的价值

**如果先测速度，结论会是"16× 参数削减 + 一个算错的层"** ——
**而"更快"的唯一原因是它算的不是原来那个东西。**

**⇒ 这是本项目第十四次同形教训的【第一次在产生任何性能数字之前】被拦住。**

#### 7.55.4 已知问题（给下一次）

1. **置换方向反了**：`forward` 里 `x[:, perm]` 作为矩阵是 `Pᵀ`，而 `effective_weight` 里
   我用的是 `P` 本身 ⇒ **两者互逆，这最可能就是 6.26 的来源**。
2. **矩形未实现**：512→2048 与 2048→2736 是 `w12`（宽 2736）与 `w3`（回 512）的真实形状，
   **不支持就替换不了 MLP**。
3. **块大小整除**：2736 = 16×171，代码里有回退循环但未验证。

#### 7.55.5 而别忘记为什么做这个

```
工程手段上限已量化: 【1.78×】（7.53 核显卸载）
而它由"CPU 剩下的 47.6%"封顶（attention 已最优 + elementwise 已融合）
⇒ ★【1.78× 到不了那道 10 倍的坎】
⇒ 所以 Monarch 不是"再快一点",而是唯一能改数量级的方向
⇒ 而 16× 的参数削减说明量级是对的
```

#### 7.55.6 顺带：又一次工具坑

**本节第一次是用 here-string 写的，三引号断了 ⇒ 没写进报告，而 git 提交成功了。**
**⇒ 与早先"inline 命令里的三引号会断"是同一条，已在笔记里 ⇒ 又踩了一次。**

**⇒ 固定做法：凡是带多行文本的补丁，一律先落成 .py 文件再执行，不用 here-string。**

'''

m = re.search(r'^### 7\.55 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.55'
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
