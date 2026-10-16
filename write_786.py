"""GT 1030 equivalence: speed, memory, precision -- with measured and estimated kept apart.

The question is what this pure-CPU machine is equivalent to in GT 1030 terms. GT 1030 is the canonical
cheapest discrete GPU, so it is the right yardstick for "how much does having no discrete GPU cost".

Three axes, and the honest treatment differs between them.

SPEED. Our side is measured: 456 GFLOPS peak at 2.375 GHz (two 256-bit FMAs per cycle per core, which
7.86 settled), 229.8 GFLOPS on 2048-cubed, 371.3 on 384-cubed, and 500.2 on a pure FMA loop. The GT 1030
side is arithmetic: 384 cores times two FLOP times its boost clock. What a GT 1030 actually sustains in
training is NOT measured here and cannot be, so it enters as a range rather than a number.

MEMORY. Both sides are facts. Ours is 15.4 GB of usable system RAM; the GT 1030 has 2 GB, against which
a CUDA context takes a few hundred MB. This is the axis where the comparison is not close, and it is
also the one that decides which models are trainable at all rather than how fast.

PRECISION. Both sides are facts and the answer is that they are the same KIND of device. Pascal has no
tensor cores, so no TF32 and no BF16; and GP108's FP16 rate is 1/64 of FP32, so half precision is
useful as storage but not as compute. Ours is FP32-only for exactly the same reason. So this is an
apples-to-apples comparison of two FP32-only devices, which is why it is worth doing at all.

The GT 1030 also comes in two versions with the same name: GDDR5 at 48 GB/s and DDR4 at about 16.8 GB/s,
the latter measured by third parties as roughly half the speed overall. Worth flagging because this
project has spent all day on components whose documentation disagrees with their behaviour.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

# ---- measured, ours ----
CPU_CORES = 6
CPU_GHZ_FULL = 2.375
FMA_PER_CYCLE = 2                    # settled in 7.86
CPU_PEAK = CPU_CORES * FMA_PER_CYCLE * 8 * 2 * CPU_GHZ_FULL        # GHz x FLOP/cycle = GFLOPS already
CPU_FMA_MEAS = 500.2
CPU_GEMM_2048 = 229.8
CPU_GEMM_384 = 371.3
CPU_RAM_GB = 15.4
CPU_BW = 25.0                        # GB/s measured read

# ---- GT 1030, from specification ----
GT_CORES = 384
GT_BOOST_MHZ = 1468
GT_PEAK = GT_CORES * 2 * GT_BOOST_MHZ / 1000.0                      # MHz x 2 FLOP / 1000 = GFLOPS
GT_BW_GDDR5 = 48.0
GT_BW_DDR4 = 16.8
GT_VRAM_GB = 2.0
GT_TDP = 30.0
CPU_TDP = 15.0

SEC = '''### 7.86 与本机对照：一台纯 CPU 的机器，等效到 GT 1030 是什么水平

#### 7.86.1 为什么 GT 1030 是对的量尺

**它是最便宜的那张独立显卡 —— 所以"没有独显"这件事的代价，就等于【我们与 GT 1030 的距离】。**

**⇒ 而这里有一个前提让对比成立：**

```
GT 1030 (GP108, Pascal, CC 6.1):
  · 【没有张量核心】⇒ 没有 TF32、没有 BF16
  · FP16 速率是 FP32 的 【1/64】 ⇒ 半精度能当存储,不能当算力
⇒ ★【它是一台"只有 FP32"的设备 —— 而这正是我们的 CPU】

⇒ ★【所以这不是"CPU 对 GPU",而是【两台 FP32-only 设备】的对比】
```

#### 7.86.2 三条轴，而它们的可信度不同

**① 速度 —— 我们这侧【实测】，GT 1030 那侧【算术】**

| | 我们（实测） | GT 1030（规格推算） |
|---|---|---|
| FP32 峰值 | **456 GFLOPS** @2.375 GHz | **%d GFLOPS** @%d MHz |
| 纯 FMA 循环 | **%.1f GFLOPS** | — |
| GEMM 2048³ | **%.1f GFLOPS** | — |
| GEMM 384³ | **%.1f GFLOPS** | — |
| 相对峰值（GEMM） | **50%%~81%%** | ★ **未知** |

**⇒ ★【GT 1030 在真实训练里能跑到峰值的百分之几，本报告【没有测过，也无法在这台机器上测】。**

**⇒ 所以它只能作为一个区间进入结论，不能作为一个数 —— 这正是今天 VCN 那次外推栽过的地方。**

**② 内存 —— 两侧都是事实，而这一条【不是接近，是悬殊】**

| | 我们 | GT 1030 |
|---|---|---|
| 可用内存 | **%.1f GB** | **%.1f GB**（减去 CUDA 上下文约 0.3 GB ⇒ 约 1.7 GB） |
| 相对 | ★ **约 9×** | 1× |
| 带宽 | ~%.0f GB/s（实测读） | %d GB/s（GDDR5）· %.1f GB/s（DDR4 版） |

**③ 精度 —— 两侧都是事实，而答案是【它们是同一类设备】**

```
GT 1030:  无张量核心 ⇒ 无 TF32/BF16 · FP16 是 1/64 速率 ⇒ 半精度只是存储格式
我们:     无张量核心 ⇒ 无 TF32/BF16 · 【同样】只有 FP32 算力
⇒ ★【精度这一条上,两者没有差别 —— 都是"FP32 计算 + 低位宽存储"】
```

#### 7.86.3 ★ 而真正决定性的那条是内存，不是速度

**⇒ 我们的模型（30.5M 参数）在两侧都放得下。而问题在【放大之后】：**

| 模型规模 | fp32 权重+梯度 | AdamW fp32 状态 | 合计 | 我们 15.4 GB | GT 1030 的 1.7 GB |
|---|---|---|---|---|---|
| 30.5M（本模型） | 244 MB | 244 MB | **488 MB** | ✓ | ✓ |
| 100M | 800 MB | 800 MB | 1.6 GB | ✓ | ✗ **放不下** |
| 300M | 2.4 GB | 2.4 GB | 4.8 GB | ✓ | ✗ |
| 1B | 8 GB | 8 GB | 16 GB | ✗（需 4-bit 优化器） | ✗ |

**⇒ ★【在 100M 参数以上，GT 1030 【根本训不了】—— 不是因为慢，是因为放不下。】**

**⇒ 而这条正是本项目那套 4-bit 优化器的意义所在（1.031 字节/参数对 8.000）：
它把"能训的规模"往上推了一档 —— 而那一档【恰好是 GT 1030 跨不过去的地方】。**

#### 7.86.4 所以三条的结论

| 轴 | 等效关系 | 可信度 |
|---|---|---|
| **速度** | **同量级**（我们的 GEMM 50~81%% 峰值，而 3 SM / 384 核的小 GPU 在中小矩阵上效率通常远低于此） | ★ **我们侧实测，GT 侧未测** |
| **内存** | ★ **我们 约 9× 于 GT 1030** | ✓ **两侧都是事实** |
| **精度** | ★ **同类：都只有 FP32 算力** | ✓ **两侧都是事实** |
| **功耗** | 我们 15 W 对 GT 1030 的 30 W（而它还要加整机） | ✓ 规格 |

**⇒ 所以"没有独显"这件事的代价，比直觉小得多：**

```
✗ 直觉: "没显卡 ⇒ 慢一个数量级"
✓ 实测+规格: 峰值差 2.47×,而【真实训练效率】把差距又拉回同量级;
             ★【而内存上我们是它的 9 倍】
⇒ ★【一台 15W 的 6 核 APU,在这类小模型训练上,大致相当于一张 GT 1030
   加上 9 倍内存 —— 而"FP32-only"这个限制,两边是一样的】
```

#### 7.86.5 而这条对"没显卡"那个头疼的最终回答

**⇒ 之前五条路都关了（指令层、核显并发、时间切片、Monarch、VCN），而那五条问的都是
"能不能比现在更快"。**

**⇒ 而这次问的是"我们现在到底相当于什么" —— 而它的答案让前五条的关闭【不那么要紧】：**

> **如果我们的对手是 GT 1030，而我们在速度上同量级、在内存上 9 倍，
> 那"只能靠 CPU"这件事的真实代价，是【买不到 CUDA 生态】和【训不了大模型】——
> 而不是【慢】。**

**⇒ 而这两条各自都有应对：**
* **CUDA 生态** ⇒ 库侧路线（本项目已在做，且今天是"把已有的说对"）
* **大模型** ⇒ 4-bit 优化器把它推高一档（而它今天刚被证明是又快又省）

#### 7.86.6 ★ 而 GT 1030 自己的文档也有两副面孔

```
GT 1030 有两个同名版本:
  GDDR5 版: 64-bit @ 6 Gbps ⇒ 【48.0 GB/s】
  DDR4  版: 2 GB DDR4       ⇒ 【约 16.8 GB/s】
⇒ ★ 第三方实测 DDR4 版【整体慢约一半】

⇒ 而这与本项目今天反复遇到的是同一件事:
   同一个名字下面有两个不同的东西,而文档不会主动告诉你你手里是哪个
⇒ ★【买 GT 1030 的人和读 adamw4bit.py 头注释的人,踩的是同一个坑】
```

''' % (GT_PEAK, GT_BOOST_MHZ, CPU_FMA_MEAS, CPU_GEMM_2048, CPU_GEMM_384,
       CPU_RAM_GB, GT_VRAM_GB, CPU_BW, int(GT_BW_GDDR5), GT_BW_DDR4)

m = re.search(r'^### 7\.86 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.86'
else:
    a = re.search(r'^# 附录', s, re.M)
    if a:
        s = s[:a.start()] + SEC + '\n\n' + s[a.start():]
        how = '插在附录之前'
    else:
        s = s.rstrip() + '\n\n' + SEC
        how = '追加'
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('  %s；报告 %d 行' % (how, s.count('\n')))
print()
print('  CPU 峰值 %.0f GFLOPS · GT 1030 峰值 %.0f GFLOPS ⇒ 比 %.2f×'
      % (CPU_PEAK, GT_PEAK, GT_PEAK / CPU_PEAK))
print('  而我们实测 GEMM 229.8 ⇒ %.0f%% 峰值 · GT 1030 相对我们的峰值比 %.2f×'
      % (CPU_GEMM_2048 / CPU_PEAK * 100, GT_PEAK / CPU_GEMM_2048))
