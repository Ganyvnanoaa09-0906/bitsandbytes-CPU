"""Append 7.98: forward and backward both on the iGPU, and the bug that a passing test did not catch.

The last piece of iGPU training, and the failure mode that makes it worth recording: every individual
check passed and the training loop still did not train.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md"
s = io.open(P, encoding="utf-8").read()

SEC = '''### 7.98 ★★★ 前向与反传都上了核显 —— 而"通过的检验"没能抓住那个 bug

#### 7.98.1 做了什么

**两处更正进了这一版：**

```
① §7.96 的反传在 CPU,而它的成本被【每次重传 50 MB 权重】主导
   ⇒ 本版把权重【常驻】核显,只上传一次（§7.97 的更正）
② §7.97 的结论:平衡是【换手】而不是【分工】⇒ 本版不切分工作,而是整条循环跑在一个引擎上
```

#### 7.98.2 ★★ 而梯度等价性【通过了】

```
前向 + 反传都在核显:
  dX  相对误差 【1.336e-06】  ✓
  dW  相对误差 【1.020e-06】  ✓
⇒ ★【Function 本身是对的】
```

#### 7.98.3 ★★★ 而真训练循环里，核显那条【不学】

```
CPU    loss 510.13 → 391.86 · 中位步时 10.1 ms · 前5步均 499.85 · 后5步均 【405.69】 ⇒ ✓ 在学
核显   loss 510.13 → 508.03 · 中位步时 28.6 ms · 前5步均 513.84 · 后5步均 【513.15】 ⇒ ✗ 几乎不动
```

**（两条臂从同一个 seed 出发，起始 loss 都是 510.13。）**

#### 7.98.4 根因：缓存键用了 `data_ptr()`

```python
wk = (int(B.data_ptr()), 'T', M)
if wk not in _wcache:
    ... 上传 ...
bb = _wcache[wk]        # ★ 之后永不上传
```

**⇒ 而 `opt.step()` 是【原地】更新 `p.data` ⇒ `data_ptr()` 不变
⇒ ★【缓存永远返回【第一步】的权重 ⇒ 核显那条臂在【对着冻结的权重】训练。】**

**⇒ 而它表现得"像在训练"：loss 从 513.84 只掉到 513.15 —— 一个很小的下降，
因为输入 x 在变（不同的 batch）、而权重不变 ⇒ 仍会有微小的 loss 波动。**

#### 7.98.5 ★★★ 而这条是今天最重要的一条方法论

```
★ 梯度等价性通过了 —— 而它是一个"看起来充分的检验"
★ 而它只测【一次前向 + 一次反向】,测不到【跨 optimizer.step() 的缓存一致性】
⇒ ★【"我验过了"这句话的含义,完全取决于【验的是什么】】

⇒ 而这一族今天已经出现多次:
   · 前向一致而反传错  ⇒ 所以验梯度
   · 梯度等价而跨步错  ⇒ ★【所以还要验"loss 真的在掉"】
   · loss 在掉而权重冻结 ⇒ 还要验"权重真的变了"（§7.94 的 mean|Δ|）
   ⇒ ★【每一层检验都盖不住下一层,而唯一知道"到哪一层为止"的办法是【跑一次真的】】
```

**⇒ 所以这个训练循环的检验要写成【级联】，而不是一个：**

| 层 | 检查 | 本次结果 |
|---|---|---|
| ① | 前向数值等价 | ✓ 7.7e-07 |
| ② | 梯度等价 | ✓ 1.3e-06 |
| ③ | **loss 真的在掉** | ★ **✗ 核显臂几乎不动** |
| ④ | 权重真的变了（mean\\|Δ\\|） | ✗ 未测（而它会立刻抓到冻结） |

**⇒ ③ 把它抓到了 —— 而如果只有 ① ②，这一条会被写成"核显训练成功，快/慢 X 倍"。**

#### 7.98.6 修法与它为什么没被修

**⇒ 修法很直接：缓存键加一个版本号（每步 +1），或每步后失效。**

**⇒ 而本轮没有修 —— 因为定位它用掉了本轮最后的上下文。**
**⇒ 而定位它【已经是这一节最有价值的部分】：一个正确的内核 + 一个正确的 Function +
一个通过的检验，而训练仍然不工作。**

#### 7.98.7 所以"核显训练"的现状

| 项 | 状态 |
|---|---|
| 前向在核显 | ✓ 等价 7.7e-07 |
| 反传在核显 | ✓ 等价 1.3e-06 |
| 权重常驻 | ✓ 已实现（§7.97 的更正） |
| **端到端训练** | ★ **✗ 缓存跨步失效，已定位，未修** |
| 速度 | 核显 28.6 ms 对 CPU 10.1 ms ⇒ ★ **仍慢 2.8 倍**（而这是在 M=1024 的两层 MLP 上，
  K=512 而 N 只有 4~512 ⇒ 形状对核显不利，见 §7.97 的 M/N 依赖） |

**⇒ 所以"核显训练"这件事的准确状态是：【组件全对，接线有一处错的，而已定位。
而它没有一个速度优势 —— 至少在这个形状组合上。】**

'''

m = re.search(r"^### 7\.98 ", s, re.M)
if m:
    nxt = re.search(r"^### |^# ", s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + "\n" + s[end:]
    how = "替换 7.98"
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
