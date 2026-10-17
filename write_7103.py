"""Append 7.103: CPU versus device-resident on the same layer. The last cell, filled honestly.

Everything else about iGPU training is verified. This measures whether any of it is faster, and on the
layer used to isolate the traffic saving, it is not.
"""
from __future__ import annotations

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

P = r"D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md"
s = io.open(P, encoding="utf-8").read()

SEC = '''### 7.103 ★★ 最后一格：同一层上，CPU 比设备常驻快 1.91×

#### 7.103.1 测量

**层 512×512 · batch 1024 · Adam · 12 步 · 两侧同样的数据与初值。**

| | 中位步时 | 均值 | 离散 |
|---|---|---|---|
| **CPU（torch 全流程）** | ★ **6.78 ms** | 7.05 ms | 10.7% |
| **设备常驻（GEMM+Adam 全在核显）** | 12.94 ms | 13.61 ms | 14.7% |

**⇒ ★【CPU 快 1.91×。】**

**而设备那条的主机流量是 6.29 MB/步（上 4.19 · 下 2.10）—— 而它的参数只有 1 MB
⇒ ★【流量确实降下来了（省掉的 5.24 MB/步不再出现），而速度没有变快。】**

#### 7.103.2 ★★★ 而这不是矛盾，而正是 §7.97 说的事

```
§7.97 的换手结论: 核显要【大 N、大 M】才赢
而这一层: N=K=512 · batch=1024 ⇒ 参数只有 1 MB、每步算术强度低
⇒ ★【核显的并行度喂不满,而它每步的固定开销（kernel 启动、传输同步、
   Python/ctypes 调用、以及 tanh 与 loss 仍在 CPU）占比就高】
⇒ ★【所以这个结果【只对这一个形状成立】,而 §7.97 已经给出了它会翻的条件。】
```

#### 7.103.3 所以核显训练这条线的完整账

| 层 | 结果 |
|---|---|
| 前向等价 | ✓ 7.7e-07 |
| 梯度等价 | ✓ 1.3e-06 |
| Adam 等价 | ✓ 1.7e-07 |
| 端到端逐位一致 | ✓ loss 与权重都一致 |
| **流量结构** | ✓ **O(NK) → O(BK+BN)**（省 5.24 MB/步） |
| **速度（本形状）** | ★ **✗ CPU 快 1.91×** |

**⇒ ★【所以"核显训练"的准确结论是:**
**它【在技术上完整成立】—— 前向、反传、优化、端到端一致、流量结构更优；
而它【在这一层上不快】，因为那一层的算术强度不够喂满核显。】**

#### 7.103.4 而这条与今天所有别的路都不同

```
§7.62   功耗仲裁            ⇒ ✗ 机制错
§7.88   缓存争抢            ⇒ ✓ 机制对
§7.89   分块并发 1.81×       ⇒ △ 未复现
§7.97   换手                ⇒ ✓ 实测
§7.99   核显训练逐位一致     ⇒ ✓ 实测
§7.102  设备常驻,流量降一档  ⇒ ✓ 实测
§7.103  ★ 而本形状上 CPU 仍快 1.91×  ⇒ ★【实测】
⇒ ★【所以这是今天唯一一条【每一层都有实测、每一层的结论都不同】的路 ——
   而不是一条被一个数字定死的路。】
```

**⇒ 而它的下一步是清楚的、有界的：把 N 和 M 往上扫（§7.97 已有 M 的曲线），
找出【什么时候设备常驻真的赢】—— 而 §7.97 说那在 M 大的时候。**

#### 7.103.5 而今天这一整轮的方法论收尾

```
✗ 把一个五组件的系统一次塞进一个脚本   ⇒ 什么都没验证成（§7.100）
✓ 每一步只验证一个东西,并按"最强检验优先"排:
     Adam 内核与 torch 逐步对拍（§7.101）
     → 然后接一层线、用流量记账当判据（§7.102）
     → 最后才测速度（§7.103）

⇒ ★【而失败的形状是同一个:【一次想验证太多】。
   而它的反面不是"更小心",是【把验证拆成互不依赖的几层,每层单独能判定】。】
```

'''

m = re.search(r"^### 7\.103 ", s, re.M)
if m:
    nxt = re.search(r"^### |^# ", s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + "\n" + s[end:]
    how = "替换 7.103"
else:
    a = re.search(r"^# 附录", s, re.M)
    if a:
        s = s[:a.start()] + SEC + "\n\n" + s[a.start():]
        how = "插在附录之前"
    else:
        s = s.rstrip() + "\n\n" + SEC
        how = "追加"
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("  %s；报告 %d 行" % (how, s.count("\n")))
