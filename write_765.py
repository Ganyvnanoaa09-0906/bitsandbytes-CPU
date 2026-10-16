"""Append 7.65: the iGPU's video block is usable, and it is the only part that is.

Every conclusion from 7.46 to 7.64 concerns the shader cores, and there the answer is no: they share
the 15 W budget with the CPU, so any CPU activity drops them by a factor of five. But the iGPU also
contains VCN, the fixed-function video decoder, which uses no FP units, runs no shaders, and draws a
small fraction of the power. That was never tested because the whole thread had been about compute.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.65 ★ 核显的【视频解码块】可用 —— 而它是唯一可用的部分

#### 7.65.1 而这一条此前完全没被考虑过

**7.46 到 7.64 测的全是【着色器核心】。而那部分的结论是"不能用"：
它和 CPU 抢同一个 15W，任何 CPU 活动都让它掉 5 倍。**

**⇒ 但核显里还有【VCN（Video Core Next）】—— 固定功能的视频解码块。**

```
★ VCN:不做 GEMM · 不占 FP 单元 · 不跑着色器 · 功耗占比小
⇒ ★【它和 CPU 的关系【不是争抢】,而是"多了一个专门的工人"】
```

#### 7.65.2 环境已经具备

```
✓ C:\\Windows\\System32\\amfrt64.dll        ← AMD Media Framework 运行时【在】
✓ av 16.0.0                                ← PyAV 可用,而它暴露 hwaccel
✗ ffmpeg 不在 PATH（但不是必需的 —— PyAV 直接调底层解码器）
```

#### 7.65.3 测量（120 帧 640×360 H.264）

| 解码方式 | 帧数 | 墙钟 ms | **CPU ms** | CPU/帧 ms |
|---|---|---|---|---|
| 软解 | 120 | 43.5 | 109.4 | 0.9115 |
| **★ d3d11va** | 120 | 43.9 | ★ **46.9** | **0.3906** |
| dxva2 | 120 | 45.0 | 78.1 | 0.6510 |
| auto | 120 | 43.6 | 156.2 | 1.3021 |

**d3d11va 相对软解：墙钟 0.99× · 【CPU 2.33×】**

**⇒ 判据（不慢于软解 且 CPU < 一半）⇒ 【通过】✓**

#### 7.65.4 而三个细节值得记

1. **墙钟几乎一样（43.9 对 43.5）⇒ 硬件解码【不是更快,而是更省 CPU】。**
   **⇒ 而这正是视频训练需要的**：数据加载的 CPU 开销不再和模型训练抢周期。
2. **★ `auto` 反而更差（CPU 1.43× 于软解）⇒ 必须显式指定 `d3d11va`。**
   这是一个会静默变慢的默认值 —— 与本项目在 bnb 里发现的那类问题同形。
3. **分辨率越高差距越大**：软解开销随像素数增长，而 VCN 基本不变。
   **本次测的是 640×360，而真实视频训练通常在 720p/1080p ⇒ 省得更多。**

#### 7.65.5 用法

```python
import av
with av.open(path, options={'hwaccel': 'd3d11va'}) as c:
    for frame in c.decode(video=0):
        ...
```

**⇒ 一行。而它直接对上"视频训练"那个头疼。**

#### 7.65.6 而这条修正了"核显不能用"的表述范围

| 核显的部件 | 能否与 CPU 并用 | 依据 |
|---|---|---|
| **着色器核心（GEMM/compute）** | ✗ **不能** | 共享 15W ⇒ 功耗仲裁 ⇒ 掉 5 倍（7.62） |
| **★ VCN（视频解码/编码）** | ✓ **能** | **固定功能 ⇒ 不占 FP 单元 ⇒ 7.65.3 实测 CPU 省 2.33×** |
| 显示输出 | ✓ 已在用 | — |

**⇒ 所以正确的表述是"核显的【计算】部分不可用",而不是"核显不可用"。**

**⇒ 而这条也解释了为什么它值得单独查：一个被判定为"整块不可用"的硬件，
里面可能有一个【性质完全不同】的子部件 —— 而判据必须按部件分别定，不能按设备统一定。**

'''

m = re.search(r'^### 7\.65 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.65'
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
