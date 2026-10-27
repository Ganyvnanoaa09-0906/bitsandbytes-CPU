"""Append 7.78: what is left, with the INT8 target now identified precisely.

The INT8 route was listed as an open item earlier but the target was vague -- 'the backend has no
integer dot-product instructions'. Checking whether it is even viable produced a much sharper answer,
and a different target than expected.
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

SEC = '''### 7.78 还剩什么 —— 而 INT8 的目标比原先写的更明确

#### 7.78.1 先问"这条路成不成立"

**此前的说法是"AVX2 后端零个整数点积指令，值 2~4.5×"。而动手前要先答一个问题：
bnb 的量化路径反量化后是 fp32 还是 int8？若是 fp32，整数路径根本用不上。**

| 检查 | 结果 |
|---|---|
| `dequantizeBlockwise4bitCpu` 的实例化 | `<float>` · `<fp16_t>` · `<bf16_t>` ⇒ ★ **全是浮点** |
| `dequantizeBlockwise8bitCpu` 的实例化 | `<float>` · `<fp16_t>` · `<bf16_t>` ⇒ ★ **全是浮点** |
| **但** cpu_ops.cpp L1125~1192 | ★ **有一个 int8 量化内核**（`int8_t* out` + `rowStats` + `threshold`） |

**⇒ 所以反量化本身走浮点，而 int8 是【另一条】路径（激活量化）。**

#### 7.78.2 ★ 而真正的缺口写在一行注释里

```
ops.py:27  # Without AVX-512 (including aarch64), torch._int_mm uses a scalar fallback
```

**⇒ ★ 缺口不是"后端没有 int8"，而是【`torch._int_mm` 在 AVX2 上退化成标量】。**

**⇒ 而 AVX2 完全能算整数点积：**

```
_mm256_maddubs_epi16   +   _mm256_madd_epi16
⇒ 这是 AVX2 的标准整数点积组合（Zen2 有；VNNI/VPDPBUSD 是 Zen4 才有）
⇒ 而全文件的直方图里【一条都没有】
⇒ 值 2~4.5×
```

#### 7.78.3 所以剩下的活，最终排序

| # | 活 | 值 | 状态 |
|---|---|---|---|
| **①** | **AVX2 int8×int8 GEMM，替 `torch._int_mm` 的标量回退** | **2~4.5×** | ★ **成立，未动** |
| **②** | 4-bit 优化器的 AVX2 路径 | 约 1% 步时 | 规格已记（7.77） |
| **③** | **网格 16×16 长训练** | ★ **26 个 epoch（对 0.43）** | **只改一个参数** |

#### 7.78.4 而 ① 与 ③ 是两种不同的东西

```
① 是【库侧的倍数】—— 让别人的模型更快,与我们训什么无关
③ 是【训练侧的数据遍数】—— 只对我们这次实验有意义

★ 而 ③ 是唯一还没试过、且【不需要写内核、不需要测硬件】的杠杆
  ⇒ 网格 32 ⇒ 1024 token ⇒ 实测 0.43 个 epoch（一个 epoch 17 小时）
  ⇒ 网格 16 ⇒  256 token ⇒ 26 个 epoch
  ⇒ ★【而"没有任何一次跑完过一个 epoch"是今天最刺眼的一条读数】
⇒ 所以 ③ 要回答的不是"能不能更快",而是【0.43 个 epoch 是不是那个真正的问题】
```

#### 7.78.5 而 ③ 的一个前提

**20+ 小时，要插电跑。**

**⇒ 本项目已经有一条纪律：不在电池上跑长任务。而这条纪律在 10-15 就是为这件事立的。**

'''

m = re.search(r'^### 7\.78 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.78'
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
