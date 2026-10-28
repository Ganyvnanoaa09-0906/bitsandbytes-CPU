"""Insert a topic index into the report, immediately after the abstract.

The report is 106 numbered sections in chronological order, which is right for the record and useless for
a reader who wants to know what was concluded about, say, the iGPU. This adds the missing axis.

Organised by question rather than by section number, and every entry carries the section that proves it
plus how far to trust it. Retractions are indexed alongside conclusions, because they are the harder-won
half: eleven sections of this report exist only to overturn an earlier one, and a reader who does not
know that will believe whichever they read first.

Inserted after the abstract so it is the first thing after the summary, before section 1.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import io
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

P = r"D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md"
s = io.open(P, encoding="utf-8").read()

ANCHOR = "## 1 引言"
if ANCHOR not in s:
    raise SystemExit("锚点没找到: %s" % ANCHOR)

INDEX = '''---

## 结论索引（按问题组织）

**本文是 106 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，
每条都指向【证据所在的那一节】，并标明可信度。**

**图例**：★★★ 多轮实测且跨测量方式一致 · ★★ 实测但只一种方法 · ★ 单次实测 · ⚠️ **曾被推翻或撤回**

### A. 这台机器到底能算多快

| 问题 | 结论 | 证据 | 可信度 |
|---|---|---|---|
| CPU 的 FP32 峰值 | **456 GFLOPS** @2.375 GHz（每周期 2 个 256-bit FMA） | §7.86 | ★★ |
| CPU 的实测 GEMM | 229.8（2048³）/ 371（L2 内） | §7.6 · §7.86 | ★★ |
| "续续轰击一个算子"的陷阱 | 会触发最严厉降频（94.87 GFLOPS），而真实训练 221~241 | 摘要 · §7.5 | ★★ |
| 内存带宽的真瓶颈 | 不是 DRAM，是 **CPU 的 outstanding-miss 能力（MLP）** | §7.3 | ★★ |
| 内存延迟 | **正常**（65 ns）—— 三个数字（259 / 119 / 65）中前两个是测量假象 | §7.69→§7.71 | ★★★ |
| 核显峰值 | 1150 GFLOPS（Vega 6，384 shader）· 实测 CLBlast **644.9** | §7.19 · §7.87 | ★★ |
| 核显 vs CPU | ★ **不是"谁快"，是"M 决定"**：M=1 给 CPU，M≥64 给核显（约 2×） | §7.97 | ★★★ |

### B. 核显到底能不能用于训练（15 节的完整弧线）

| 问题 | 结论 | 证据 | 可信度 |
|---|---|---|---|
| 为什么 CPU 一忙核显就掉 | ★ **共享 8 MB L3 被 CPU 工作集挤掉，不是功耗仲裁** | §7.88 | ★★★ |
| 那为什么是"悬崖" | ⚠️ **曾经的答案"开关式功耗仲裁"是错的** —— 我扫了线程数，没扫工作集 | §7.62→§7.88 | ⚠️ |
| 两者能不能一起干 | ★ **只多 7%** —— 共享 L3 是【零和的】 | §7.97 | ★★★ |
| 正确的形态 | ★ **换手**：小 M 用 CPU，大 M 用核显 | §7.97 | ★★★ |
| 设备常驻训练 | ★ **与 CPU【逐位一致】**，流量 O(NK)→O(BK+BN) | §7.99 · §7.102 | ★★★ |
| 而它快吗 | ★ **只在【参数≥16 MB 且 K 大】时赢 1.30×**；512×512 上 CPU 赢 1.84× | §7.103 · §7.104 | ★★★ |
| 核显的【非计算】部件 | ★ **VCN 视频解码可用**（固定功能，不占 FP 单元）—— 而它的收益后来被否（⚠️ §7.73） | §7.65 · ⚠️§7.73 | ★ / ⚠️ |

### C. 库侧：bitsandbytes 上什么变了

| 问题 | 结论 | 证据 | 可信度 |
|---|---|---|---|
| 4-bit 优化器是工具还是玩具 | ★ **是工具**：30.5M 模型 300 步，loss 差 −0.10% | §7.76 | ★★ |
| 而它快吗 | ★ **快 2.63×**（CI 2.13~3.12），不是我头注释写的"慢 4×" | §7.80 | ★★★ |
| 它省多少内存 | **对 fp32：7.8×** · ★ **对 8-bit：2.06×**（而后者才是用户的对手） | §7.75 · §7.80 | ★★ |
| 而它该在哪用 | ★ **参数 ≥ 300M** —— 30.5M 上只省 1.3% 内存 | §7.76 · §7.94 | ★★ |
| 4-bit GEMV 的缺口 | ⚠️ **"AVX2 静默回退"是错的** —— AVX2 融合路径【早已建好、接好、阈值 64】 | §7.59→§7.74 | ⚠️ |
| 那 10.6× 是什么 | 是【Python 层阈值从 16 提到 64】的收益，不是"补实例" | §7.74 | ★★ |
| INT8 GEMM | ★ **写完、6 形状逐位验证** · 对 fp32 朴素 21~72× · 对 torch：**M=1 赢 3.05×，M=128 输 2.36×** | §7.105 | ★★★ |
| 而 int8 快不快 | ★ **取决于 M**（与核显同构）· 而 **AVX2 的 24% 指令效率接近结构上限**（无 VNNI） | §7.105 · §7.106 | ★★★ |
| ⚠️ **分块能救 int8 吗** | ⚠️ **不能** —— 实现后 0.91×。诊断错了：它 ALU 受限，不是访存受限 | §7.106 | ⚠️ |

### D. 训练加速：什么是真的

| 问题 | 结论 | 证据 | 可信度 |
|---|---|---|---|
| Muon vs AdamW | **1/3.17 的步数达同一 loss**；而算上开销净 **1.89×** | §7.27 · §7.33 | ★★ |
| 权重衰减 | ★ **项目至今最大的一次质量提升** | §7.38 | ★★ |
| 非矩阵乘算子占多少 | **3~9%** —— 所以只优化 GEMM 是对的 | §7.12 | ★★ |
| batch 有效吗 | **无收益** —— 开销由 token 数决定 | §7.20 | ★★ |
| torch.compile | **不成立，−2.4%** | §7.42 | ★ |
| LoRA 训练（Tiny-SD） | 管线通（6.74 s/步），而**模型质量下降**（48 张 + 恒定 caption ⇒ 过拟合） | §7.92 · §7.94 | ★★ |
| 冻结主体 + 训适配器 | ★ **训练步 ≈ 推理步** —— 因为反向【跳过】冻结参数的梯度 | §7.92 | ★★ |

### E. 生成质量：天花板在哪

| 问题 | 结论 | 证据 | 可信度 |
|---|---|---|---|
| 瓶颈在分词器吗 | **不** —— 分词器往返近无损（码本 1024/1024 用满） | §7.48 | ★★ |
| 那在哪 | ★ **在采样，不在预测**：教师强制距自身天花板 1.6 dB，自由采样塌陷 6.8 dB | 摘要 | ★★ |
| token 数 vs 码本大小 | ★ **token 数扩 4× 值 3.5~3.9 dB，码本扩 8× 只值 0.54 dB** | 摘要 | ★★ |
| MaskGIT | ⚠️ **实现完整而从未被执行过** —— 第一次真跑是 10-15 深夜 | §7.47 | ⚠️ |
| 课程学习 | **无效，且机制清楚** | §7.30 | ★ |
| Monarch 分解 | 自检通过（10~28× FLOPs 减少），而**壁钟只 0.81~1.30×** | §7.55 · §7.56 | ★★ |

### F. ⚠️ 被撤回的结论清单（**读之前必须知道**）

**本报告的 106 节里有 11 节存在的唯一目的就是推翻更早的一节。若不知道这件事，会相信先读到的那一个。**

| 被撤回的 | 撤回它的 | 为什么错 |
|---|---|---|
| "核显慢于 CPU"（§7.13） | §7.19 | 分母是 DirectML 与自写 kernel 的效率，不是芯片能力 |
| "分块并发 1.81×"（§7.89） | §7.97 | 在 M=2048 上测的，而真实形状 M=128 |
| **"功耗仲裁，开关式"（§7.62）** | **§7.88** | **扫了线程数而没扫工作集 ⇒ 把缓存效应读成开关** |
| "L2 分块 1.359×"（§7.52） | §7.54 | 它是**内存分配**的假象，不是缓存 |
| "延迟 259 ns"（早期） | §7.69→§7.71 | Python 开销 + TLB |
| "浮点延迟偏高 1.6×"（§7.70） | §7.71 | 把 TLB 算成了 DRAM |
| "AVX2 静默回退，10.6×"（§7.59） | §7.74 | 查调用图后发现**不缺实例** |
| "M=2~4 缺口值 10~14×"（§7.72） | §7.74 | 阈值【早就是 64】 |
| "4-bit 优化器慢 4×"（头注释） | §7.80 | **方向相反**，实测快 2.63× |
| "没有 AVX2 路径"（§7.77） | §7.81 | `optimizer_4bit_blockwise_avx2` 在 L3056 |
| **"VCN 硬解可用"（§7.65/7.66）** | **§7.73** | **没扫分辨率；而解码本来就不是瓶颈** |

### G. 方法学：反复出现的失效模式

| 模式 | 出现次数 | 典型 |
|---|---|---|
| **用代理量代替真实约束，而它从不报错** | 摘要记 15 次，今天又添多次 | 事件 |
| 真实测量量错了对象 | 7+ | §7.95（入队当执行）· §7.96（每步传权重当常态） |
| 扫错了自变量 | 3 | §7.62（线程数 vs 工作集）· §7.69（工作集 vs TLB） |
| 从代码形状推断"缺什么" | 3 | §7.59 · §7.72 · §7.77（**都不缺**） |
| 结论反复 | 4 | 核显四次翻转；内存三次 |
| 通过一个检验就宣布成立 | 1 | §7.98（梯度等价通过，而训练没在训练） |

**⇒ 而唯一被反复证明有效的对策只有一条：**

> **把自变量扫出来。它不是测量卫生的建议，而是唯一能把混在一起的机制拆开的手段
> —— 悬崖 ⇒ 开关（功耗仲裁的误读）；两条斜率 ⇒ 两个机制（L3 跨越 + TLB）；
> 非单调 ⇒ 只能实测建表（内核分派）。**

### H. 可交付的东西（截至 2026-10-17）

| 项 | 状态 | 位置 |
|---|---|---|
| bitsandbytes CPU 补全（AVX2 融合内核） | ✓ 已发布 | `bitsandbytes/csrc/` |
| 4-bit 优化器（AdamW4bit） | ✓ 已发布 `0.50.2.dev3` | `bitsandbytes/optim/adamw4bit.py` |
| 2/3-bit 量化 | ✓ | `quant2bit.py` · `quantnbit.py` |
| GDN 反向 | ✓ 728 s → 秒级（~45×） | `csrc/cpu_gdn.cpp` |
| Muon + NS=3 | ✓ 接入训练路径 | `train_ar_v2.py` |
| 核显训练（设备常驻） | ★ **技术上成立、与 CPU 逐位一致**；适用区间【参数≥16 MB 且 K 大】 | §7.104 |
| INT8 GEMM 内核 | ★ **逐位验证；M=1 赢 3.05×**；未接进 torch | `int8_gemm_avx2.cpp` |
| 文档（中英各四份） | ✓ | `docs_cpu/` |

---

'''

s = s.replace(ANCHOR, INDEX + ANCHOR, 1)
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("  ✓ 结论索引已插入（在摘要之后、§1 之前）")
print("  ✓ 报告 %d 行" % s.count("\n"))
