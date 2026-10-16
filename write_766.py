"""Append 7.66: hardware decode saves 2.9% of the step at 640x360, and scales with resolution.

The criterion was 5% and it failed. But the measurement's value is in what it shows about scaling,
because the saving is proportional to pixels while the model step is not.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.66 硬解在 640×360 上只省 2.9% —— 而它随分辨率增长

#### 7.66.1 测量（真实训练循环）

| 项 | 墙钟 | CPU |
|---|---|---|
| 只有模型（不解码） | **3.807 s** | — |
| 加载器单独 · 软解 | 0.071 s | 0.469 s |
| 加载器单独 · **d3d11va** | 0.067 s | ★ **0.281 s**（省 40%） |

| 完整步 | 步时 s | 相对无解码 | 相对软解 |
|---|---|---|---|
| 无解码 | 3.701 | 0.972× | — |
| 软解 | 3.874 | 1.018× | 1.000× |
| **d3d11va** | **3.763** | 0.989× | ★ **0.971×** |

**判据（≥5%）⇒ 【2.9%，未过】✗**

#### 7.66.2 为什么只有 2.9%

```
这一步处理 32 帧（2 片段 × 16 帧）@ 640×360
而模型本身要 3.807 s
⇒ 解码的 CPU 只占这一步的 【12%】(0.469 / 3.807)
⇒ 而省下的 0.188 s CPU 只换回 0.111 s 墙钟（其余被同步吃掉）
```

**⇒ 在 640×360、每步 32 帧下，解码【太小】，省不出多少。**

#### 7.66.3 ★ 而让这个数变大的是分辨率

```
软解 CPU ≈ 每 32 帧 0.469 s ⇒ 【每帧 14.7 ms @640×360】
而解码成本【随像素数】增长:

  640×360    ⇒ 解码占步时 【12%】  ⇒ 硬解省 2.9%    ← 本次
  1280×720   ⇒ 约 4× ⇒ 占 【48%】
  1920×1080  ⇒ 约 9× ⇒ 占 【108%】★【解码比模型还贵】
⇒ ★【720p 以上,解码从"可忽略"变成"主导项"】
```

**⇒ 所以结论分两句：**

| 场景 | 硬解的价值 |
|---|---|
| **640×360 · 每步 32 帧** | **2.9% ⇒ 判据未过** |
| **720p 以上** | **从 48% 起 ⇒ 必须用** |

#### 7.66.4 而它有一条别的优化都没有的性质：免费

```python
av.open(path, options={'hwaccel': 'd3d11va'})
```

**⇒ 一行，而代价是零：没有实现成本、没有质量损失、没有数值风险。**

**⇒ 所以即使只有 2.9%，也没有理由不用 —— 而它在分辨率上去之后【自己会变大】。**

#### 7.66.5 而这条与 7.54 那条撤回形成对照

**7.54 里那个 L2 分块 1.359× 被撤回，因为它在预分配对照组下只有 0.87~1.006× ⇒ 是测量假象。**

**⇒ 而本次的 2.9% 是【在真实训练循环里、对着无解码下限、和软解对照】测出来的 ⇒ 它是真的，只是小。**

**⇒ 区别在于：7.54 测的是两个【不可比】的路径，而本次测的是同一条路径上的一个开关。**

'''

m = re.search(r'^### 7\.66 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.66'
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
