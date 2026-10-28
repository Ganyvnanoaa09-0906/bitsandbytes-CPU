"""Correct TECH_REPORT.md's iGPU section and add the components it predates.

This is the results report and its section 3.7 still carries the position that the iGPU is a per-operator
regression with block residency the only viable form. That position has been revised twice since. A
results report that leaves a superseded conclusion standing is worse than one that omits it, because the
conclusion reads as current.

So 3.7 gets a revision block rather than a rewrite: the original reasoning and its measurements stay
readable, and what changed is stated next to it. And 3.9 is added for the components the report predates
-- the 4-bit optimizer and 2/3-bit quantisation -- so the results document is complete rather than only
correct.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import io
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

P = r"D:\work\bitsandbytes-CPU\docs_cpu\TECH_REPORT.md"
s = io.open(P, encoding="utf-8").read()

# ---------- 1) §3.7 后面加修订块 ----------
ANCHOR = "## 4. 生图训练实测（R5-4500U）"
if ANCHOR not in s:
    raise SystemExit("锚点没找到: %s" % ANCHOR)

REV = '''### 3.8 ★★ 对 §3.7 的两次修订（2026-10-17）

**§3.7 的结论是"逐算子负优化 → 块级常驻可行"。此后又做了两轮测量，而两轮都改变了那个结论——
而方向不同，所以两轮都记在这里。**

#### 3.8.1 第一次修订：机制不是功耗，是共享 L3

**此前把"CPU 一忙核显就掉 5×"归因为【15W 封装里的功耗仲裁】。而那个归因是错的。**

实测（线程数固定为 1，只改 CPU 的工作集）：

| CPU 负载（1 线程） | 工作集 | 核显相对空闲 |
|---|---|---|
| 纯 ALU 自旋（无访存） | 0 | ★ **1.03×** |
| `torch.mm` 256³ | 0.8 MB | 1.96× |
| `torch.mm` 1024³ | 12 MB | 3.76× |
| `torch.mm` 2048³ | 48 MB | ★ **6.08×** |

**⇒ ★【一个满功耗、零访存的负载只让核显掉 3%，而工作集越大掉得越多
⇒ 机制是【共享的 8 MB L3 被 CPU 的工作集挤掉】。】**

**⇒ 而这条修订的意义不是学术性的：**
* 功耗仲裁是【开关式】的 ⇒ 只能靠加功率绕开 ⇒ 而本机读不到温度也无从监控 ⇒ 不可接受
* 缓存争抢是【连续】的 ⇒ 它【可调】：压小 CPU 的工作集就能把它买回来

#### 3.8.2 第二次修订：不是分工，是换手

**扫 M（N=6144, K=2048，真实层的形状）：**

| M | 核显单独 | CPU 单独 |
|---|---|---|
| 1 | 5.3 | ★ **11.7** |
| 64 | ★ **356.8** | 56.7 |
| 2048 | ★ **425.3** | 199.7 |

**⇒ ★【M=1 给 CPU，M≥64 给核显（约 2 倍）。
而"两者一起"只比核显单独高 7% —— 因为共享 L3 是【零和的】。】

**⇒ 所以"分工"这个形态是错的，正确的是【换手】：小 M 用 CPU，大 M 用核显，
而不要让它们同时干同一份活。**

#### 3.8.3 ★★★ 而设备常驻训练已完成，并与 CPU 【逐位一致】

**§3.7 说"块级常驻可行"。现在可以更进一步，而结论是【有条件的】：**

```
✓ OpenCL GEMM 作为 torch.autograd.Function,前向与反传都在核显
✓ 梯度等价:dX 1.3e-06 · dW 1.0e-06（对 torch autograd）
✓ Adam 内核:与 torch 逐步对拍 1.7e-07;转置副本【逐位精确】
✓ ★ 端到端训练与 CPU 【逐位一致】(loss 轨迹与权重变化量完全相同)
✓ 每步主机流量 O(NK) → O(BK+BN)
```

**→ 而适用区间是被测过的：**

| N × K × B | 参数 | CPU ms | 设备 ms | 判定 |
|---|---|---|---|---|
| 512×512×1024 | 1.0 MB | 7.24 | 13.28 | ★ **CPU 赢 1.84×** |
| 1024×1024×1024 | 4.2 MB | 40.68 | 37.69 | 平 |
| **2048×2048×1024** | 16.8 MB | 152.56 | **117.42** | ★ **设备赢 1.30×** |
| 4096×1024×4096 | 16.8 MB | 570.55 | 594.87 | 平 |

**⇒ ★【适用区间 = 【参数 ≥ 16 MB 且 K 大】。而 batch 几乎无影响（1024→4096 时 1.30→1.31）。】**

#### 3.8.4 ★ 一条使用提醒，因为这条线上量错过四次

```
入队时间当执行时间   ⇒ 高估 40 倍（46787 GFLOPS，而设备峰值 1150）
每步重传权重当常态   ⇒ 低估 5.8 倍（64.5 对 374 GFLOPS）
```

**⇒ ★【判据：任何超过设备标称峰值的核显数字，先当测量错误处理，而不是当发现。】**

**⇒ 而在小层/小 batch 上，核显执行器【不该开】—— 判据是【形状】，不是"有没有核显"。**

---

## 3.9 ★ 本轮新增的两个组件

### 3.9.1 4-bit 块优化器（AdamW4bit）

| 对手 | 内存 | 速度 |
|---|---|---|
| 对 fp32 AdamW | **7.8×** 少（1.031 对 8.000 字节/参数） | 0.96× |
| **对 AdamW8bit** | ★ **2.06×** 少 | ★ **2.63× 更快** |

**⇒ ★【两边都赢，不是"用速度换内存"。而早期文档写的"没有 AVX2 路径、慢 4×"两句都是错的。】**

**⇒ ★ 而它的战场是【大模型】：30.5M 参数上只省 1.3% 内存，300M 上省 13%。**

### 3.9.2 2-bit / 3-bit 量化

**`quant2bit.py`（2-bit 专用）· `quantnbit.py`（通用 N-bit，2/3/4/5 位，带自检往返）·
`quant_lora.py`（接入训练）· `linear2bit.py` / `embed2bit.py`（层替换）。**

**★ 而一条重要的区分：降【权重】精度已被否定（§3.5），而降【优化器状态】精度是正面的。
两者改的不是同一个东西，所以"4-bit 好"这个说法必须带主语。**

---

'''

s = s.replace(ANCHOR, REV + ANCHOR, 1)
print("  ✓ §3.8 修订块 + §3.9 新组件 已插入")

# ---------- 2) §7 结论里加一条 ----------
OLD7 = "## 7. 结论与展望"
NEW7 = '''## 7. 结论与展望

> **2026-10-17 补充：关于核显的那条结论请以 §3.8 为准。**
> 机制是共享 L3（不是功耗），形态是换手（不是分工），
> 而设备常驻训练已与 CPU 逐位一致、适用区间为【参数 ≥ 16 MB 且 K 大】。
> **在小层上它的结论是【负的】**（512×512 时 CPU 快 1.84×）。'''
if OLD7 in s:
    s = s.replace(OLD7, NEW7, 1)
    print("  ✓ §7 结论已加指向 §3.8 的提示")

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("  ✓ TECH_REPORT.md 现在 %d 行" % s.count("\n"))
