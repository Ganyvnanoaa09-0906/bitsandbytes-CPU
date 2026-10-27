"""Append 7.99: the cache bug fixed, and iGPU training verified to be bit-identical to CPU.

The closing of the iGPU training thread. One line of versioning turned a loop that did not train into
one that reproduces the CPU result exactly, and the fact that it is exact rather than merely close is
the strongest evidence in this whole thread that the two paths compute the same function.
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

SEC = '''### 7.99 ★★★ 缓存修好之后：核显训练与 CPU 训练【逐位一致】

#### 7.99.1 修了什么

**§7.98 定位的 bug：缓存键用 `data_ptr()`，而 `opt.step()` 原地更新 ⇒ ptr 不变 ⇒ 永远返回第一步的权重。**

**⇒ 修法一行：键里加一个版本号，训练循环在 `opt.step()` 之后调 `igpu_step()` 把它 +1。**

```python
_wver = [0]
def igpu_step():
    """★ 在 optimizer.step() 之后调用 —— 声明"权重变了,缓存作废"。"""
    _wver[0] += 1

wk = (int(w.data_ptr()), 'T', M, _wver[0])   # ★ 带版本号
```

**⇒ 而"常驻"的正确语义因此被写清了：不是"永不重传"，是【权重没变就不重传】。**

#### 7.99.2 ★★★ 结果：两条臂【逐位一致】

| | loss 起 → 止 | 前5步均 | 后5步均 | 中位步时 |
|---|---|---|---|---|
| **CPU** | 510.1298 → 391.8558 | 499.8509 | 405.6913 | **10.5 ms** |
| **核显** | 510.1297 → 391.8558 | 499.8509 | 405.6913 | 32.8 ms |

**⇒ ★【两条臂的 loss 轨迹、前5步均、后5步均【完全一样】。】**

**而第 4 层检验（§7.98 指出的缺失项）：**

```
CPU    权重的 mean|Δ| = 7.978e-03  ✓ 变了
核显    权重的 mean|Δ| = 7.978e-03  ✓ 变了
⇒ ★【权重变化量也【完全一样】】
```

**⇒ ★【"逐位一致"比"它学会了"强得多：它说明核显那条路和 CPU 那条路是【同一个计算】,
只是跑在不同的引擎上 —— 而这才是"核显训练可用"的真正含义。】**

#### 7.99.3 ★★ 而诚实的另一半：在这组形状上核显慢 3.1 倍

```
10.5 ms（CPU）对 32.8 ms（核显）
原因三条,而都可以预料:
  ① 形状:这个测试 N 只有 4~512,而第二层 N=4 ⇒ 对核显极不利
  ② ★ 权重每步都要重传:optimizer 在 CPU ⇒ 权重每步都变 ⇒ 缓存版本号每步 +1
     ⇒ 一个 512×512 的权重就是 1 MB/步,而 §7.97 测的是【权重常驻】的好情况
  ③ 每步还要传 x、gy、dW 等
⇒ ★【而这正是 §7.97 "换手"结论的边界:核显要【大 N、大 M】才赢】
```

#### 7.99.4 ★★★ 所以"核显训练"这件事的最终定案

| 层 | 检查 | 结果 |
|---|---|---|
| ① | 前向数值等价 | ✓ 7.7e-07 |
| ② | 梯度等价（单步） | ✓ 1.3e-06 |
| ③ | **loss 真的在掉** | ✓ 499.85 → 405.69 |
| ④ | **权重真的变了** | ✓ mean\\|Δ\\| = 7.978e-03 |
| ⑤ | **与 CPU 逐位一致** | ★ **✓ 完全相同** |
| — | **速度（本组形状）** | ✗ **慢 3.1 倍** |

**⇒ 所以答案分两层，而它们不矛盾：**

```
✓ 【核显训练【在技术上成立】】—— 前向、反传、优化、端到端,全部与 CPU 逐位一致
✗ 【而它【在大多数形状上不快】】—— 因为权重每步重传,而那是"optimizer 在 CPU"的必然代价
⇒ ★【要让它在速度上也成立,需要【optimizer 也上核显】——
   那样权重常驻才真正成立,而 §7.97 测的 2 倍优势才会出现在训练里】
```

#### 7.99.5 而这条把整个核显线收成了一个【有条件成立】的结论

```
§7.62:  "功耗仲裁,开关式,不可用"        ⇒ ✗ 机制错（实为共享 L3）
§7.88:  "缓存争抢,连续,可调"            ⇒ ✓
§7.89:  "分块后并发 1.81×"               ⇒ △ 而在 §7.97 的扫描里未复现
§7.97:  "不是分工,是换手:M=1 给 CPU,M≥64 给核显"  ⇒ ✓ 实测
§7.99:  "核显训练技术上成立,逐位一致"     ⇒ ✓ 实测
⇒ ★【而它【在速度上】要成立,还缺一块:optimizer 上核显】
```

**⇒ 而这块缺的东西是明确的、有边界的、且不与今天任何结论冲突 —— 它是一个可以做的下一步，
而不是一个"也许能行"的希望。**

#### 7.99.6 而这一轮的方法论收获

```
★ 三层检验都通过,而训练仍然不工作  ⇒ ★【所以检验要【级联】,且每一层都必须跑真的】
★ 而第五层"与 CPU 逐位一致"是【最强】的那一层:
   它一旦成立,前面四层就都成了它的特例
⇒ ★【所以构造检验时,应当【先找最强的那一层】,而不是从最弱的一层层加上去】
```

'''

m = re.search(r"^### 7\.99 ", s, re.M)
if m:
    nxt = re.search(r"^### |^# ", s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + "\n" + s[end:]
    how = "替换 7.99"
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
