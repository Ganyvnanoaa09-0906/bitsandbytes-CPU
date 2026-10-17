"""Append 7.58: the INT8 microbenchmark says 4.5x, and it is measuring the wrong thing.

Two findings, one about the kernel and one about the measurement. The INT8 inner loop is 4.5x the
FP32 one, which is more than the 2x predicted from instruction counts -- and the reason is that both
loops are bound by cache bandwidth, not by instruction throughput, so the 4x smaller data type is
what shows up rather than the 2x fewer instructions.

The measurement was designed to avoid that: operands were kept cache-resident precisely so the
comparison would be about instructions. It did not work, because at 128 KB per iteration for FP32
the loop moves 65 GB/s, which is L2 territory, and it is the traffic that limits both.
"""
from __future__ import annotations

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.58 INT8 微基准：4.5×，而它量的不是它声称要量的东西

#### 7.58.1 结果

用 `D:\\vs\\VC\\Tools\\Llvm\\x64\\bin\\clang++.exe`（clang 22.1.3）编译，`-O2 -mavx2 -mfma`，
两个内层循环都作用在缓存驻留的数据上（N=16384，4000 次重复）：

| 实现 | GFLOP/s | 相对 |
|---|---|---|
| **FP32 FMA（8 宽）** | **16.7** | 1.000× |
| **INT8 `maddubs` + `madd_epi16`** | **75.0** | ★ **4.501×** |

**⇒ 而两个累加值都【是对的】**：

```
sf = 4000.0 而期望 = REPS × rf = 4000 × 1.0 = 4000 ✓
si = -40000 而期望 = REPS × ri = 4000 × (-10) = -40000 ✓
```

**⇒ 脚本里报"算错"是【我自己的检查写错了】—— 拿单次的 `rf` 去比累加了 4000 次的 `sf`。**
**⇒ 又一次同形错误：判据本身没错，但它比的不是同一个东西。**（第十五次）

#### 7.58.2 但 4.5× 不是"一条指令干两件事"

**按指令数预测是 2×，而实测 4.5× ⇒ 说明主因不是指令吞吐。**

```
FP32 循环: 每次迭代读 128 KB（两个 64 KB 数组）
           tf ≈ 7.8 s ⇒ 带宽 ≈ 【65 GB/s】← L2 量级
INT8 循环: 每次迭代读  32 KB（两个 16 KB 半宽）
           ti ≈ 1.75 s ⇒ 带宽 ≈ 【146 GB/s】

⇒ ★【两个循环都是【缓存带宽受限】,不是指令受限】
⇒ 4.5× 的主要来源是【int8 数据小 4 倍】,不是"指令少一半"
```

**⇒ 所以这个微基准的 4.5× 是【带宽密集场景】的数字。**

#### 7.58.3 而我们的 GEMM 不是那个场景

```
GEMM 算术强度 ≈ 300 FLOP/字节,而带宽受限门槛 ≈ 9
⇒ 高出 【33 倍】⇒ 【计算受限】（7.52.2 已确立）
⇒ 所以它拿到的会是【指令层面的约 2×】,不是 4.5×
```

**⇒ 老实说：这个微基准测出了一个真实的 4.5×，但它量的是带宽密集场景，而我们的场景不是那个。**

#### 7.58.4 而设计上的错在哪

**我【刻意】把数据放进缓存，就是为了排除带宽因素 ⇒ 目标是对的，做法没达到目标：**

```
N=16384 floats = 64 KB/数组 ⇒ 128 KB/迭代
⇒ 那【确实】在 L2 里（3 MB），但每字节只做了 2 次 FLOP
⇒ ⇒ 而 L2 的带宽（65 GB/s 量级）不足以喂满 456 GFLOPS 的 FMA
⇒ ★【"放进缓存"不等于"不受带宽限制" —— 还要看每字节做多少 FLOP】
```

**⇒ 正确的做法是让内层循环在【寄存器】里复用（比如把每个加载值用 8 次），
那才能把两者的差距压到纯指令层面。**

#### 7.58.5 结论没变，但依据要换

| 说法 | 依据 | 状态 |
|---|---|---|
| INT8 有真实收益 | ✓ | **成立**（4.5× 带宽密集 / 约 2× 计算密集） |
| 对我们训练的 GEMM 值 2× | 指令数推算 | ✓ 仍是推算，**未直接测** |
| 对训练整体 | — | **1.36%，且与核显卸载重叠 ⇒ 天花板仍是 1.78×** |

**⇒ 而这条的定位不变：它是【库级】的真实收益（尤其对带宽密集的推理 GEMV），
而不是这个项目训练的突破口。**

'''

m = re.search(r'^### 7\.58 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.58'
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
