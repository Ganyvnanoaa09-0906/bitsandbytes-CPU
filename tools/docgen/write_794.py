"""Append 7.94: the LoRA training run completed, and the honest reading of its output.

The pipeline works end to end. The model it produced is worse than the one it started from, for reasons
that were predicted before the run and are worth stating as plainly as the fact that it ran.
"""
from __future__ import annotations

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.94 ★★ LoRA 训练跑通了 —— 而它的产出比原模型更差，原因可以预料

#### 7.94.1 实测

```
150 步 · 总计 1049.4 s（17.5 分钟）· 中位步时 6.74 s
★ 权重实际变化量 mean|Δ| = 【2.103e-03】
LoRA 已保存 → tinysd_lora/lora

训练前出图 4 张: 120.0 s（每张 30 s）
训练后出图 4 张: 121.8 s

★ 像素级差异（同 prompt 同 seed）:
  astronaut  平均差 12.45 · 最大差 161.0 · 变化像素 【64.2%】
  city       平均差 30.06 · 最大差 194.0 · 变化像素 【99.5%】
  portrait   平均差 19.98 · 最大差 233.0 · 变化像素 【97.7%】
  landscape  平均差 23.04 · 最大差 130.0 · 变化像素 【93.4%】
```

#### 7.94.2 ★★★ 而看图之后，结论与数字不同

| | 训练前 | 训练后 |
|---|---|---|
| **city** | 干净扁平矢量城市 · 边缘锐利 · 橙蓝分明 · 楼体清晰 | ★ **模糊、灰雾、去饱和 · 楼体糊成一片** |
| | | ★ **顶部出现【乱码文字伪影】（"IRV… DUV…"）** |

**⇒ ★【像素差 99.5% 是真的 —— 而它是【退化】,不是学到的风格。】**

#### 7.94.3 而原因全都可以预料，而且我在跑之前就写下了

**脚本结尾那句：**

```
★ 如实预期: 48 张图 · 恒定 caption · 150 步 ⇒ 这学的是【风格倾向】,不是概念。
   能问的不是"好不好",而是【权重动了没有、输出跟着动了没有】。
```

**⇒ 而答案分三句：**

```
① 权重动了没有    ⇒ ✓ 动了（mean|Δ| = 2.1e-3）
② 输出跟着动了没有 ⇒ ✓ 动了（64~99.5% 像素变化）
③ 动得好不好      ⇒ ✗ 【更差】
```

**⇒ 而"更差"的三个原因，每一个都在跑之前就能列出：**

```
· 数据集 48 张 ⇒ 150 步 × batch1 ÷ 48 ≈ 【3 个 epoch 的同一批图】⇒ 过拟合
· caption 恒定 ⇒ 适配器只能学"这批图的平均风格",而学不到"什么条件对应什么"
· 乱码文字伪影 ⇒ ★【模型被推到了训练分布之外】,而不是学到了新分布
```

#### 7.94.4 ★ 所以这一步的产出是什么

**⇒ 不是"一个更好的模型"，而是【一条端到端跑通的训练管线】+【一组可复现的数】：**

| 项 | 值 | 状态 |
|---|---|---|
| 训练步时 | **6.74 s**（150 步 · 17.5 分钟） | ✓ 实测，与推理步同量级 |
| 前向/反向构成 | 注意力份额 14.6% → **25.5%** | ✓ §7.92 |
| LoRA 可训练占比 | **0.863M / 324.2M = 0.27%** | ✓ |
| 出图 | 4 步 · 512×512 · 每张 30 s | ✓ 可用 |
| **模型质量** | ★ **比原模型差** | ✗ **而原因已知** |

**⇒ ★【"跑通了"和"跑得好"是两个不同的问题，而这一节把两个都答了 ——
而不是用第一个的好消息去盖住第二个的坏消息。】**

#### 7.94.5 而让它变好需要什么（如果以后要做）

```
① caption 要逐图（这个集合没有 ⇒ 换一个带文本的数据集）
② 图要多（48 → 数百张）
③ 步数要按 epoch 算而不是按步数拍（150 步 ≈ 3 epoch ⇒ 1~2 epoch 就够）
④ ★ 而这台机器的约束是【出图 30 s/张 · 训练 6.74 s/步】——
   ②③ 的成本因此是线性的，可算
⇒ ★【所以"做不做得好"不是一个未知问题,是一个【多花多少小时】的问题】
```

'''

m = re.search(r'^### 7\.94 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.94'
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
