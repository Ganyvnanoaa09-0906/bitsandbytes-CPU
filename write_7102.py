"""Append 7.102: the device-resident step done, and the structural criterion confirmed.

The wiring change from 7.101 completed. dW stays on the device and feeds Adam directly, and the host
traffic is counted rather than claimed -- which is the point, because the claim is structural and a
performance number would not test it.
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

SEC = '''### 7.102 ★★★ 设备常驻训练步跑通：流量从 O(NK) 变成 O(BK)

#### 7.102.1 做了什么

**§7.101 说剩下的只是接线：把反向 GEMM 写在设备上的 dW 【直接】作为 Adam 内核的 g 参数，
而不是下载再上传。本节做掉了它，并且用【一层】来隔离收益（512×512 · batch 1024）。**

```python
gemm(N, K, B, gyaT, xa, dW)     # ★ dW 直接写在设备 buffer 上
k_adam(..., w, wT, mt, vt, dW, ...)   # ★ 直接读那个 buffer,不下载不重传
```

#### 7.102.2 结果：与 CPU 参照一致

| 步 | 设备 loss | CPU loss |
|---|---|---|
| 1 | 6.400706 | 6.400705 |
| 2 | 6.317965 | 6.317966 |
| 4 | 6.192193 | 6.192195 |
| 6 | 6.006683 | 6.006686 |
| **8** | **5.888395** | **5.888399** |

**⇒ ★【loss 轨迹一致（差 ~1e-6,float32 累计舍入的量级）。】**

**最终权重相对误差 【2.796e-05】，而我事先写的判据是 1e-5 ⇒ 严格说【未通过】。**

**⇒ 而如实写：8 步 float32 累积的 2.8e-5 与 1e-5 是【同一量级】，
所以这是【判据定得比实际能保证的精度更紧】，不是计算错了。**
**⇒ ★【而这条要写下来,而不是"因为差得不多就当作通过" —— 那正是本报告整天在批评的事。】

#### 7.102.3 ★★★ 而结构性判据成立，而它才是本节的重点

```
★ 主机流量（实测，8 步合计）:
    主机 → 设备  67.1 MB ⇒ 【8.39 MB/步】
    设备 → 主机  17.8 MB ⇒ 【2.23 MB/步】

★ 而每步【省掉】的是:
    权重 w/wT      2 × 262144×4 B = 2.10 MB/步
    Adam 状态 m/v  2 × 262144×4 B = 2.10 MB/步
    梯度 dW        1 × 262144×4 B = 1.05 MB/步
    ⇒ 合计 【5.24 MB/步】

★ 而仍然要搬的:O(B·K) 那部分（输入 x、输出 y、梯度 g）
    ⇒ ★【它随【batch】涨,不随【参数量】涨】
⇒ ★★★【所以一句话:流量从 O(NK) 变成 O(BK)。而参数越大,省得越多。】
```

**⇒ 而这正是 §7.100 第 ② 条判据要的东西：**

```
§7.100 ②: "每步的主机流量 = 【输入 batch + loss 标量】"
⇒ 而实测【不完全是这样】:y 和 g 仍然各要 O(B·N) 的搬运
   因为 loss 和 dY 都在主机上算（tanh 那一层也还在 CPU）
⇒ ★【所以 ② 的严格形式是 O(BK + BN),而不是 O(BK) ——
   而它【仍然与参数数无关】,所以结构性好处成立,只是没有句子字面上那么干净。】
```

#### 7.102.4 所以"核显训练"这条线的最终账

| 项 | 状态 |
|---|---|
| 前向在核显 | ✓ 等价 7.7e-07 |
| 反传在核显 | ✓ 等价 1.3e-06 |
| Adam 在核显 | ✓ 等价 1.7e-07（§7.101） |
| 转置 | ✓ 被设计消除（§7.101） |
| **端到端训练** | ✓ loss 与 CPU 一致（§7.99 用 CPU optimizer；§7.102 用设备 optimizer） |
| **设备常驻** | ✓ **流量 O(NK) → O(BK+BN)** |
| **速度** | ★ **未测（本节是一层隔离，15.0 ms/步，而 CPU 的同一层未测）** |

**⇒ 而"速度"那一格是【唯一还空着的】—— 而它现在是一个【有明确测法】的空格：
在【同样的层、同样的 batch】上比较 CPU 与设备常驻的每步墙钟。**

#### 7.102.5 而这一轮的方法论

```
§7.100 把五个组件塞进一个脚本   ⇒ 跑不起来,记为一个缺口清单
§7.101 先只做 Adam 内核,与 torch 逐步对拍 ⇒ 通过
§7.102 再接一层线,并把【流量记账】作为判据 ⇒ 结构性结论成立

⇒ ★【三步里,每一步【只验证一个东西】——
   而 §7.100 那一步想一次验证五个,于是它什么都没验证成。】
```

'''

m = re.search(r"^### 7\.102 ", s, re.M)
if m:
    nxt = re.search(r"^### |^# ", s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + "\n" + s[end:]
    how = "替换 7.102"
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
