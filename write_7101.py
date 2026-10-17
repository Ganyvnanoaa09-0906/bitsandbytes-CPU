"""Append 7.101: the Adam kernel done, and the transpose kernel designed away.

Of the two kernels 7.100 listed as missing, one is written and verified and the other turned out not to
be needed. The gap to device-resident training is now wiring, not new code.
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

SEC = '''### 7.101 ★★★ Adam 内核写完并验证，而转置内核被【设计消除】

#### 7.101.1 而这一轮的做法与 7.100 那次失败相反

```
§7.100 的失败:把转置、Adam、两个方向的权重、梯度布局、kernel 参数顺序
            全塞进一个脚本 ⇒ 三处死分支、一个前向引用的函数 ⇒ 跑不起来

§7.101 的做法:★【先只做【一个】内核,而它【单独可验证】——
             与 torch 的 Adam 逐步对拍,对上了再接进任何循环】
⇒ ★【而这正是"最强的那一层检验优先"那条(§7.99)的具体用法:
   一个优化器内核的错误会产出【照常训练而静默偏差】的模型,
   所以它必须先被单独钉死。】
```

#### 7.101.2 ★★ 而转置内核被一个设计选择消掉了

```
观察:
  · 前向需要 w 的 [K,N] 布局（y = x @ w^T 里 w^T 是 [K,N]）
  · 反向需要 w 的 [N,K] 布局（dX = gy @ w 里 w 是 [N,K]）
⇒ 存【两份】,而让 Adam 内核【在同一个 kernel 里更新两份】:
     int n = i / K; int k = i - n * K;
     p[i] -= lr * mh / (sqrt(vh) + eps);
     pT[k * N + n] = p[i];        // ★ 同一趟里写转置副本
⇒ ★【转置内核不需要了 —— 而它的代价只是一个【倒数索引】,
   不是一趟【完整的内存遍历】】
```

**⇒ 所以 §7.100 列的两个缺失内核，现在【一个写完、一个不存在】。**

#### 7.101.3 ★ 验证：与 torch 逐步对拍 5 步

| 步 | 与 torch 的相对误差 | 转置副本误差 |
|---|---|---|
| 1 | 2.905e-08 | **0.000e+00** |
| 2 | 5.817e-08 | **0.000e+00** |
| 3 | 1.162e-07 | **0.000e+00** |
| 4 | 1.160e-07 | **0.000e+00** |
| 5 | 1.737e-07 | **0.000e+00** |

**⇒ ★【Adam 内核与 torch 一致（1.7e-07，float32 的累积舍入量级），而转置副本【逐位精确】。】**

**★ 而对拍的参照是【手写的同一个公式】,不是 torch.optim.AdamW ——
因为要验的是【这一个内核的语义】,而不是"两边都调库"。**

#### 7.101.4 而 Adam 内核的吞吐是访存受限的

```
512×512（262144 元素）· 0.782 ms ⇒ 【8.0 GB/s 有效带宽】
每元素:5 个数组读（p/pT/m/v/g）+ 5 个写,而只做约 10 次浮点运算
⇒ ★【所以它是【带宽受限】的,和本项目的其它 elementwise 算子同族
   （§7.94 那组:elementwise/搬运占 11.8~13.6%）】
⇒ 而 512×512 是 1 MB 的状态 ⇒ L3 之内 ⇒ 8 GB/s 偏低,
  说明它【没有跑满】—— 而那是一个可以单独优化的点,不是本次的目标
```

#### 7.101.5 ★★★ 所以"完全设备常驻"还差什么：只剩【接线】

**⇒ §7.100 列的缺口清单，现在的状态：**

| 组件 | §7.100 的状态 | 现在 |
|---|---|---|
| GEMM 内核（前向/反向） | ✓ 有 | ✓ |
| **转置内核** | ✗ 要写 | ★ **不需要了（设计消除）** |
| **Adam 更新内核** | ✗ 要写 | ★ **✓ 写完并验证** |
| 两方向权重常驻 | ✗ 要接 | ✗ 要接 |
| Adam 的 m/v 常驻 | ✗ 要接 | ✗ 要接 |
| **dW 留在设备上直接喂 Adam** | （7.100 未列） | ★ **这是真正剩下的那一步** |

**⇒ 而最后一条是【接线】,不是新内核：**

```
现在:  反向的 dW 是 GEMM 的输出 ⇒ 它在设备上（一个 buffer）
       而 §7.101 的验证是【把 dW 从主机传进去】
⇒ 差的一步:把那个 buffer 【直接】作为 Adam 内核的 g 参数传进去
⇒ ★【一行参数的改动,而它让每步的主机流量从 O(N·K) 降到 O(B·K)】
```

#### 7.101.6 所以核显训练这条线的最终状态

```
§7.62   功耗仲裁               ⇒ ✗ 机制错
§7.88   缓存争抢               ⇒ ✓
§7.89   分块并发 1.81×          ⇒ △ §7.97 未复现
§7.97   不是分工是换手          ⇒ ✓ 实测
§7.99   核显训练逐位一致        ⇒ ✓ 实测
§7.100  速度需设备常驻,缺 2 内核 ⇒ 缺口清单
§7.101  ★ 内核全齐（1 个新写、1 个消掉）,★ 只剩【接线】
⇒ ★【而这是今天唯一一条【从"概念成立"推进到"只差接线"】的路。】
```

'''

m = re.search(r"^### 7\.101 ", s, re.M)
if m:
    nxt = re.search(r"^### |^# ", s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + "\n" + s[end:]
    how = "替换 7.101"
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
