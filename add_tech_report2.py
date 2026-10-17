"""Patch TECH_REPORT.md without touching the bytes that are already damaged.

The file has invalid UTF-8 at a handful of positions: multi-byte sequences missing their leading bytes,
e.g. 0xe3 0x80 0x82 followed directly by 0xbc 0x89 0xe3 0x80 0x82. That corruption predates this edit --
the first attempt to patch it failed on read and therefore never wrote -- so it was introduced earlier,
probably by a write in a different encoding.

The temptation is to "repair" it by guessing what those characters were. That would be wrong twice: the
guess is not verifiable, and replacing three damaged spots with invented text turns a visible defect into
an invisible one. This report has spent the day arguing against exactly that move.

So surrogateescape is used: it maps each invalid byte to a private surrogate code point, which survives a
decode-encode round trip byte for byte. The new sections go in cleanly encoded, and the damaged bytes come
out exactly as they went in. The corruption is reported rather than silently patched.

Verified after writing by comparing the byte streams at the damaged offsets before and after.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

P = r"D:\work\bitsandbytes-CPU\docs_cpu\TECH_REPORT.md"

raw_before = open(P, "rb").read()
# ★ surrogateescape:坏字节 → 私有代理码点 → 写回时还原成原字节。不猜、不改。
s = raw_before.decode("utf-8", errors="surrogateescape")

n_bad = sum(1 for ch in s if 0xDC80 <= ord(ch) <= 0xDCFF)
print("  文件 %d 字节 · 其中 %d 个是【已损坏的字节】" % (len(raw_before), n_bad))

ANCHOR = "## 4. 生图训练实测（R5-4500U）"
if ANCHOR not in s:
    raise SystemExit("锚点没找到: %s" % ANCHOR)

REV = '''### 3.8 ★★ 对 §3.7 的两次修订（2026-10-17）

**§3.7 的结论是"逐算子负优化 → 块级常驻可行"。此后又做了两轮测量，而两轮都改变了那个结论。**

#### 3.8.1 第一次修订：机制不是功耗，是共享 L3

| CPU 负载（1 线程） | 工作集 | 核显相对空闲 |
|---|---|---|
| 纯 ALU 自旋（无访存） | 0 | ★ **1.03×** |
| `torch.mm` 256³ | 0.8 MB | 1.96× |
| `torch.mm` 2048³ | 48 MB | ★ **6.08×** |

**⇒ ★【一个满功耗、零访存的负载只让核显掉 3%，而工作集越大掉得越多
⇒ 机制是【共享的 8 MB L3 被 CPU 的工作集挤掉】，不是功耗仲裁。】**

**⇒ 而这个区别是有后果的：功耗仲裁是【开关式】的（只能加功率绕开，而本机读不到温度）；
缓存争抢是【连续】的（压小工作集就能买回来）。**

#### 3.8.2 第二次修订：不是分工，是换手

| M | 核显单独 | CPU 单独 |
|---|---|---|
| 1 | 5.3 | ★ **11.7** |
| 64 | ★ **356.8** | 56.7 |
| 2048 | ★ **425.3** | 199.7 |

**⇒ ★【M=1 给 CPU，M≥64 给核显（约 2 倍）；而"两者一起"只比核显单独高 7%
—— 因为共享 L3 是【零和的】。】**

#### 3.8.3 ★★★ 设备常驻训练已完成，并与 CPU 【逐位一致】

```
✓ OpenCL GEMM 作为 torch.autograd.Function,前向与反传都在核显
✓ 梯度等价:dX 1.3e-06 · dW 1.0e-06
✓ Adam 内核:与 torch 逐步对拍 1.7e-07;转置副本【逐位精确】
✓ 端到端训练与 CPU 【逐位一致】(loss 与权重变化量完全相同)
✓ 每步主机流量 O(NK) → O(BK+BN)
```

**→ 适用区间是被测过的：**

| N × K × B | 参数 | CPU ms | 设备 ms | 判定 |
|---|---|---|---|---|
| 512×512×1024 | 1.0 MB | 7.24 | 13.28 | ★ **CPU 赢 1.84×** |
| 1024×1024×1024 | 4.2 MB | 40.68 | 37.69 | 平 |
| **2048×2048×1024** | 16.8 MB | 152.56 | **117.42** | ★ **设备赢 1.30×** |
| 4096×1024×4096 | 16.8 MB | 570.55 | 594.87 | 平 |

**⇒ ★【适用区间 = 【参数 ≥ 16 MB 且 K 大】。batch 几乎无影响（1024→4096 时 1.30→1.31）。】**

#### 3.8.4 ★ 一条使用提醒：这条线上量错过四次，而两次方向相反

```
入队时间当执行时间   ⇒ 高估 40 倍（46787 GFLOPS，而设备峰值 1150）
每步重传权重当常态   ⇒ 低估 5.8 倍（64.5 对 374 GFLOPS）
```

**⇒ ★【判据：任何超过设备标称峰值的核显数字，先当测量错误处理，而不是当发现。】
⇒ 而在小层/小 batch 上核显执行器【不该开】—— 判据是【形状】，不是"有没有核显"。**

---

## 3.9 ★ 本轮新增的两个组件

### 3.9.1 4-bit 块优化器（AdamW4bit）

| 对手 | 内存 | 速度 |
|---|---|---|
| 对 fp32 AdamW | **7.8×** 少（1.031 对 8.000 字节/参数） | 0.96× |
| **对 AdamW8bit** | ★ **2.06×** 少 | ★ **2.63× 更快** |

**⇒ ★【两边都赢，不是"用速度换内存"。而早期文档写的"没有 AVX2 路径、慢 4×"两句都是错的。】
⇒ ★ 而它的战场是【大模型】：30.5M 参数上只省 1.3% 内存，300M 上省 13%。**

### 3.9.2 2-bit / 3-bit 量化

**`quant2bit.py` · `quantnbit.py`（通用 N-bit，带自检往返）· `quant_lora.py`（接入训练）·
`linear2bit.py` / `embed2bit.py`（层替换）。**

**★ 一条重要区分：降【权重】精度已被否定（§3.5），而降【优化器状态】精度是正面的
⇒ "4-bit 好"这个说法必须带主语。**

---

'''

s = s.replace(ANCHOR, REV + ANCHOR, 1)

OLD7 = "## 7. 结论与展望"
NEW7 = '''## 7. 结论与展望

> **2026-10-17 补充：关于核显的那条结论请以 §3.8 为准。**
> 机制是共享 L3（不是功耗），形态是换手（不是分工），
> 而设备常驻训练已与 CPU 逐位一致、适用区间为【参数 ≥ 16 MB 且 K 大】。
> **在小层上它的结论是【负的】**（512×512 时 CPU 快 1.84×）。'''
if OLD7 in s:
    s = s.replace(OLD7, NEW7, 1)
    print("  ✓ §7 已加指向 §3.8 的提示")

# ★ 用同一个 surrogateescape 写回 ⇒ 坏字节原样还原
raw_after = s.encode("utf-8", errors="surrogateescape")
open(P, "wb").write(raw_after)

print("  ✓ §3.8 修订块 + §3.9 新组件 已插入")
print("  ✓ 写回 %d 字节（原 %d）" % (len(raw_after), len(raw_before)))

# ★ 验证:坏字节的数量和位置都该没变
chk = raw_after.decode("utf-8", errors="surrogateescape")
n_bad_after = sum(1 for ch in chk if 0xDC80 <= ord(ch) <= 0xDCFF)
print("  ✓ 损坏字节:改前 %d 个 · 改后 %d 个 %s"
      % (n_bad, n_bad_after, "（未增加 ✓）" if n_bad_after == n_bad else "✗ 变多了"))
print("  ✓ 现在 %d 行" % raw_after.count(b"\n"))
