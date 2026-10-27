"""Append 7.100: the device-resident design, stated and not completed.

A section recording what the next step requires and why it was not done in this round. Written because
the alternative -- leaving a half-written script that looks finished -- is the exact failure mode this
report has spent the day finding in other people's code.
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

SEC = '''### 7.100 设备常驻训练：设计与未完成的原因

#### 7.100.1 缺的那一块，从 §7.99 直接推出来

```
§7.99: 核显训练与 CPU 【逐位一致】,而慢 3.1 倍
⇒ 而慢的原因是结构性的:optimizer 在 CPU ⇒ 权重每步都变 ⇒ 每步都要重传
⇒ ★【所以修法不是"更快的传输",是【取消传输】:
   把权重、Adam 状态、梯度全部放在设备上，让每步的主机流量只剩【输入 batch 和 loss 标量】】
```

#### 7.100.2 而它需要的组件是清楚的

| 组件 | 状态 |
|---|---|
| GEMM 内核（前向） | ✓ 已有（§7.96/7.97 验证过） |
| GEMM 内核（反向，两个方向） | ✓ 同一个内核，换形状即可 |
| **转置内核**（[K,N] ↔ [N,K]） | ✗ **要写** |
| **Adam 更新内核**（就地更新 p/m/v） | ✗ **要写** |
| 两个方向的权重都常驻 | ✗ **要接**（前向要 [K,N]，反向要 [N,K]，存两份省一次转置） |
| Adam 的 m/v 常驻 | ✗ 要接 |
| 偏差修正（b1^t, b2^t） | ✓ 主机侧算好传进去即可 |

**⇒ 而 Adam 内核的形状很简单：**

```c
__kernel void adam(const int n, __global float* p, __global float* m, __global float* v,
                   const __global float* g, const float lr, const float b1, const float b2,
                   const float eps) {
    int i = get_global_id(0);
    if (i >= n) return;
    float mi = b1*m[i] + (1-b1)*g[i];
    float vi = b2*v[i] + (1-b2)*g[i]*g[i];
    m[i] = mi; v[i] = vi;
    p[i] -= lr * (mi/(1-b1)) / (sqrt(vi/(1-b2)) + eps);
}
```

#### 7.100.3 而这一轮【没有完成它】，原因如实写

**⇒ 我把转置、Adam、两个方向的权重、梯度的三种布局、以及每个 kernel 的参数顺序
全塞进一个脚本里赶完了 —— 而写出的是：**

```
· 三处 if False 死分支
· 一个函数在定义之前就被调用
· 一个辅助函数直接 return None
⇒ ★【那个脚本跑不起来,而它不是"再改改就好"】
```

**⇒ ★【按本项目今天反复的纪律：半个脚本比没有更糟 ——
因为一个看起来完成的脚本,会让下一个人以为这块已经做过了。】**

**⇒ 所以这一节记录【设计 + 缺口清单】，而不是记录一个假装的完成。**

#### 7.100.4 而它的验收判据是先写好的

**⇒ 设备常驻训练做成了的标志不是速度，是【三条同时成立】：**

```
① 与 CPU 训练【逐位一致】(loss 轨迹 + 权重变化量)
   ⇒ 而那已经在 §7.99 里用"optimizer 在 CPU"的版本达成过 ⇒ 是一个【可复现的基线】
② ★ 每步的主机流量 = 【输入 batch + loss 标量】
   ⇒ 而那可以直接【量出来】：用 cl.enqueue_copy 的字节数计数
③ ★ 速度上超过 CPU
   ⇒ 而 §7.97 说:权重常驻时核显在 M≥64 是 CPU 的约 2 倍
   ⇒ 所以 ② 成立之后 ③ 应当出现;若 ② 成立而 ③ 不出现,那是一条新的发现
```

**⇒ 而这三条的价值在于：② 是一个【结构性】的判据（不是性能数字），
所以它一旦成立，③ 就成了一个关于这台机器的事实陈述，而不是一个调优结果。**

#### 7.100.5 而这一条给整条核显线收尾

```
§7.62  功耗仲裁（开关式,不可用）      ⇒ ✗ 机制错,实为共享 L3
§7.88  缓存争抢（连续,可调）          ⇒ ✓
§7.89  分块后并发 1.81×               ⇒ △ §7.97 的扫描里未复现
§7.97  不是分工是换手: M=1 给 CPU, M≥64 给核显  ⇒ ✓ 实测
§7.99  核显训练技术上成立,逐位一致     ⇒ ✓ 实测
§7.100 ★ 而【速度上成立】需要设备常驻 —— 设计已定,组件缺两个内核,而本轮未完成
```

**⇒ 所以这条线的准确状态是：【概念上通了,实现上差一步,而那一步的缺口清单是明确的。】**

**⇒ 而它与今天所有别的路都不同：它是【唯一一条在概念上已经成立、只是没写完】的路。**

'''

m = re.search(r"^### 7\.100 ", s, re.M)
if m:
    nxt = re.search(r"^### |^# ", s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + "\n" + s[end:]
    how = "替换 7.100"
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
