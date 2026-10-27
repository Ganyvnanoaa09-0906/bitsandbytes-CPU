"""Define the north-star goal: training speed AND precision equivalent to a GT 1030.

The report already contains the pieces scattered across 7.86 (what a 1030 is), 7.87 (our iGPU is the
same 384-shader class, anchored at 55-60% cuBLAS efficiency), 7.109 (real-shape CLBlast 371-655),
7.110 (fp16 pure compute 1290 = 1.14x the 1030's theoretical 1127) and index row "device-resident
single-layer step 1.50x CPU, about 2x from the 1030". What was missing is a *decision procedure*:
what exactly must be measured on which workload, and what number counts as "equivalent".

This section writes that procedure down and adds the first scoreboard harness (gt1030/), so every
later optimisation is graded against the same ruler instead of against a vibe.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW_SECTION = '''### 7.118 ★★★ 北极星目标：训练速度与精度"等效 GT 1030"——可算的定义、判决矩阵与攻击路线

#### 7.118.1 为什么必须先写"判决式"

**"速度等效 GT 1030" 这句话如果不落到【哪个 workload、哪个数字、谁的时间】，它就不是一个目标，
而是一句口号。** §7.86 用 1030 当量尺（没独显的代价 = 与 1030 的距离），§7.87 把它收窄成
"我们核显与它同为 384 shader、cuBLAS 效率同区间（55~60%）"，§7.110 已经把 fp16 纯运算打到
**1290 GFLOPS = 1030 理论值 1127 的 1.14×**，但**端到端训练步**仍差约 2~5×。所以目标要拆成两条
可判定的式子：

**（速度）** 对每个命名 workload W（同模型、同 batch、同优化器）：
```
T_ours(W)  ≤  T_1030(W)  =  max( FLOPs(W) / (1127 GFLOPS × η_c),  Bytes(W) / (48 GB/s × η_m) )
```
其中 η_c = 0.56（§7.87：我们自己的 384-shader GPU 实测 cuBLAS 类效率 55~60%，与 GP108 同宽，
按同一区间取保守值）、η_m = 0.85（GDDR5 典型可达）。1030 的 PCIe 3.0 x4（约 3.9 GB/s/向）
只在"每步重传权重"的坏实现里才计入——**而 1030 的 2 GB 显存恰好让这种坏实现成为它的常态**。

**（精度）** 同一训练配方下，loss 曲线与最终质量与 fp32 CPU 基线等效（容差：loss 差 ≤1%、
生成质量同档）。允许 fp16 存储 + fp32 累加——**这并不降低要求**：1030 是 Pascal 消费卡，
FP16 只有 1/64 速率（约 17.6 GFLOPS），它自己训练时也只能 fp32 计算；我们用 fp16 存储 + fp32
累加，精度不低于 1030 能用的精度。

**⇒ ★【等效 = 同一个 workload 上，我们的墙钟时间 ≤ 1030 的 roofline 时间，且精度不输。】**

#### 7.118.2 参考模型与已测数字（全部来自本报告，不引入外部跑分）

| 量 | GT 1030（GDDR5） | 本机现状（实测） | 比值 |
|---|---|---|---|
| FP32 峰值 | 1127 GFLOPS | CPU 456（§7.86） | 0.40× |
| FP16 峰值 | **17.6**（1/64，Pascal 消费卡） | **核显 1290**（§7.110） | **73×** |
| 可达 GEMM | ~631（1127×0.56，锚定我们同宽 GPU 的实测效率） | 核显 fp16 GEMM 1039（大形状）· 真实形状 371~655 | 0.59~1.65× |
| 显存带宽 | 48 GB/s GDDR5 | DDR4 共享，约 25~35 GB/s 实配 | 0.5~0.7× |
| 容量 | 2 GB | 15.4 GB（+ 核显共享） | 7.7× |
| 主机链路 | PCIe 3.0 x4 ≈ 3.9 GB/s | 统一内存，无 PCIe 瓶颈 | — |

#### 7.118.3 判决矩阵：哪些 workload 可能，哪些不可能

以 AR-30.5M 训练步为例（§7.12 的 375 GFLOP/步，T=256×batch8=2048 token）：

| workload | 1030 roofline 时间 | 我们当前 | 1030 分数 | 判决 |
|---|---|---|---|---|
| AR-30.5M 训练步（375 GFLOP） | **0.59 s**（375/631） | CPU 2.854 s（194 GFLOPS 实配） | **0.21×** | 核显若整步跑到真实形状 371~655 ⇒ 0.57~1.01 s ⇒ **0.58~1.03×：可能打平** |
| 大 GEMM（4096³ 类） | — | 核显 fp16 1039 vs 可达 631 | **1.65×** | **已超过**（算力不是瓶颈） |
| Tiny-SD LoRA 步（6.74 s/步管线，§7.92） | 需按 SD 计算量折算（scoreboard 待测） | CPU 6.74 s | 待算 | 受带宽与 Conv 主导，**最可能卡住的 workload** |
| SD1.5 512px / fp32 严格 | — | — | — | **不可行**（1030 自身 2 GB 放不下 SD1.5 全量，我们也受 DDR4 带宽限制） |

**⇒ ★ 结论分两半：**
- **GEMM 主导的小模型训练（AR-30.5M、Tiny-SD）⇒ 有可能打平甚至超过 1030**——因为我们的核显
  fp16 算力是它 FP32 峰值的 1.14×，而它只有 55~60% 效率；
- **fp32 严格精度 / 大模型 / 512px 扩散 ⇒ 不可能**，除非：① 全程 fp16 存储+fp32 累加并验证精度；
  ② 用核显 GEMM 把 CPU 从矩阵乘里解放；③ 用内核融合把 DDR4 带宽需求压到 1030 的 48 GB/s 以下。

#### 7.118.4 攻击路线（按杠杆排序）

| 阶段 | 做什么 | KPI |
|---|---|---|
| **P0 计分板** | `gt1030/` 工具：定义 workload、1030 roofline、测 CPU/核显/常驻三种模式的步时间，输出"1030 分数" | 三 workload 的分数表（先 baseline） |
| **P1 整步在核显** | 不改模型结构，把 AR 训练步整体搬到核显（fp16 存储+fp32 累加，权重/梯度/优化器状态常驻）；CLBlast 真实形状从 371~655 提到 ≥700 | AR 步 ≤1.0 s（0.59×1030 分数 ≥0.59） |
| **P2 融合与重叠** | GEMM epilogue 融合 bias/gelu/残差；attention 留在 CPU（实测 116 GFLOPS 的算子不搬）；消除每步 H2D/D2H | AR 步 ≤0.7 s（分数 ≥0.84） |
| **P3 精度等效验证** | 同一随机种子跑 fp32 基线与 fp16 存储两条 loss 曲线，2000 步；解码图片对比 | loss 差 ≤1%，质量同档 |
| **P4 端到端** | Tiny-SD LoRA 步 6.74 s → 目标 ≤1030 折算时间；512px 只作"不可行"证据 | 两个 workload 的分数，公开 |

#### 7.118.5 第一里程碑（本轮已落地）

`gt1030/` 三个文件：
- `ref.py` —— 1030 参考常量 + roofline 判决式（可复算，不含经验黑箱）；
- `workloads.py` —— W1 AR-30.5M 步、W2 真实形状 GEMM 表、W3 Tiny-SD 步的 FLOPs/字节定义；
- `scoreboard.py` —— 测 CPU / 核显 / 常驻三种模式，打印"1030 分数"；`--report-only` 只用本报告
  已测数字，`--measure` 跑真机（等 BOS 训练跑完再跑，避免 CPU 争抢污染数字）。

#### 7.118.6 诚实边界

1. **GT 1030 不在本机，无法实测**——参考时间来自"同宽度 GPU 的实测效率 55~60%"外推，不是跑分；
2. **1030 的 2 GB 显存**决定了超过 ~100M 参数它根本训不了（§7.86.4），所以"等效"的 workload
   必须限定在它也能跑的范围内；
3. **精度等效 ≠ 数值相同**：允许 fp16 存储 + fp32 累加，但必须用 loss 曲线与样本质量验证。



'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW_SECTION + '# 附录 A：AR 训练路径的实测证据', 'insert 7.118'),
    ('**本报告的 117 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。',
     '**本报告的 118 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。', 'F header: 118'),
    ('**本文是 117 节的【时序记录】', '**本文是 118 节的【时序记录】', 'header: 118'),
    ('| **generic x86-64 第三产物** | ✓ **144 KB · AVX 指令 0 条 · 便携 selftest 5/5 · 三级选路** | `build_linux.sh --generic` · §7.117 |',
     '| **generic x86-64 第三产物** | ✓ **144 KB · AVX 指令 0 条 · 便携 selftest 5/5 · 三级选路** | `build_linux.sh --generic` · §7.117 |\n'
     '| **GT 1030 等效目标（北极星）** | ★ **定义+判决矩阵+计分板；AR 步 1030 分数 0.21× → 目标 ≥0.84** | §7.118 · `gt1030/` |',
     'H table: gt1030 row'),
]


def main() -> int:
    text = io.open(REPORT, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        if out.count(old) != 1:
            print('ABORT %-20s anchor=%d' % (label, out.count(old)))
            return 1
        out = out.replace(old, new, 1)
        print('ok    %s' % label)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(out)
    print('report chars %d -> %d, lines %d -> %d'
          % (len(text), len(out), text.count('\n') + 1, out.count('\n') + 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
