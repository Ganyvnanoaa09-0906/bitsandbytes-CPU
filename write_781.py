"""Append 7.81: 7.77 is retracted. The AVX2 path exists and has all along.

7.77 specified how to write optimizer_4bit_blockwise_avx2, on the premise stated in adamw4bit.py that
"the kernel currently has no AVX2 path". The kernel is there, the dispatcher uses it, and the comment
explaining the design is in the source. The premise was never checked.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.81 ★ 撤回 7.77：AVX2 路径【早就存在】

#### 7.81.1 cpu_ops.cpp 里的四个优化器内核

```
L2743  optimizer_8bit_blockwise_scalar
L2899  optimizer_4bit_blockwise_scalar
L3056  ★【optimizer_4bit_blockwise_avx2】   ← ★ 它【在】
L3361  optimizer_8bit_blockwise_avx2
```

**⇒ ★【4-bit 和 8-bit 【都有】 AVX2 版本。】**
**⇒ 而 `adamw4bit.py` 说 "the kernel currently has no AVX2 path" —— 那是错的。**

#### 7.81.2 而分派器会用它

`cpu_ops.cpp:3707` 的 `optimizer_update_4bit_blockwise_cpu`：

```cpp
// AVX2 只覆盖 2 状态家族（adam / ademamix）；1 状态家族走标量。
// 新代码面因此小得多，而 AdamW4bit 用的 adam 正好被覆盖。
const bool two_state = (state2 != nullptr);
bool use_avx2 = false;
```

**⇒ 所以：**

```
① optimizer_4bit_blockwise_avx2 【存在】（L3056）
② 分派器【按 two_state 走 AVX2】—— 而 adam 是 2 状态家族 ⇒ ★【会用到】
③ ⇒ "没有 AVX2 路径"这句话,从内核、分派、到设计注释,三处都不支持它
```

**★ 而那句注释本身是一个【刻意的设计决策】（AVX2 只覆盖 2 状态家族，缩小新代码面）——
⇒ 所以这不是"漏了"，而是"选了"。**

#### 7.81.3 所以 7.77 撤回

| 版本 | 内容 | 状态 |
|---|---|---|
| **7.77** | "照 optimizer_8bit_blockwise_avx2 写一个 4-bit 版" | ✗ **撤回：那个函数【已经存在】** |
| **7.77** | "值约 1% 步时" | ✗ **无关：没有活要干** |
| **7.81** | ★ **AVX2 路径已建、已接、已注释；只有 Python 包装器的头注释没跟上** | ✓ |

#### 7.81.4 而这一条与 7.80 合起来，说明了问题的真正所在

```
内核侧:  ✓ AVX2 路径存在且被使用
分派侧:  ✓ 按 two_state 正确分派,且有设计注释
基准侧:  ✗ 判据错了(用离散度代替标准误)
基准侧:  ✗ 搬运量表错了(8-bit 少算一个状态)
包装器:  ✗ ★【头注释说"没有 AVX2 路径"、"慢 4×" —— 两句都错】
```

**⇒ ★【问题不在内核里,而在【描述内核的那段文字】里。】**

**⇒ 而这段文字的代价是可以量化的：**

```
它说"慢 4×"      ⇒ 实测快 2.63×   ⇒ 方向相反,差 11 倍
它说"没有 AVX2"   ⇒ 实测有并已用    ⇒ 假的
它说"能忍受就试试" ⇒ 实测两边都赢    ⇒ 劝退
⇒ ★【任何读了它的人都不会用这个优化器 —— 而它是这个项目唯一"两边都赢"的东西】
```

#### 7.81.5 而这是今天第二次【我给已完成的活写了规格】

| 次 | 我写的规格 | 实际 |
|---|---|---|
| **7.72** | "M=2~4 的缺口值得填" | ✗ 早已填（阈值是 64） |
| **7.77** | "写 optimizer_4bit_blockwise_avx2" | ✗ 早已存在（L3056） |

**⇒ 两次的根源完全一样：**
**我从【一个形状】（模板只覆盖 bf16、文档说没有 AVX2）推出了【一个空缺】，而没有查。**

**⇒ 而在 7.74 里我已经把这条写成教训了 —— 然后又犯了两次。**

**⇒ 所以"教训写下来"不等于"行为改了"。而这一条比今天任何技术结论都重要。**

#### 7.81.6 所以真正该做的四件事

```
① ★ 改 adamw4bit.py 的头注释 —— 删掉"没有 AVX2 路径"和"慢 4×"两句
   （它们现在是错的,且方向相反,且代价是劝退使用者）
② 改 bench_opt_4bit.py 的判据（L115）—— 用标准误
③ 改搬运量表（L123-124）—— 8-bit 算两个状态
④ ★ 而"写新内核"这件事【不做】—— 没有活要干
```

**⇒ ①②③ 都是【改文字和判据】，不需要动内核。而它们的总收益
（让一个"快 2.63× 且省 2.06× 内存"的东西被正确地描述）**大于今天任何一条内核优化**。**

'''

m = re.search(r'^### 7\.81 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.81'
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
