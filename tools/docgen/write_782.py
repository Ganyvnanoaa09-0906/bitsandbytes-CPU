"""Append 7.82: the fixes are in, and the pattern generalises into an audit.

Three descriptions were corrected and verified. Five assets today were mis-described, and four of the
five were described wrongly in the code's own comments -- which makes checking those comments a
concrete, bounded piece of work rather than a lesson.
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

SEC = '''### 7.82 修正已落地并验证；而"错误描述"这件事可以变成一项审计

#### 7.82.1 三处修正（都不动内核）

| # | 位置 | 错 | 改 |
|---|---|---|---|
| **a** | `adamw4bit.py` 头注释（**两个副本**） | "no AVX2 path" · "about 4x slower" | ★ **实测：AVX2 路径在 L3056、分派器 L3707 会用它、快 2.63×** |
| **b** | `bench_opt_4bit.py:115` 判据 | `abs(med-1) > 2*pstdev` | ★ **改用 95% CI 是否排除 1** |
| **c** | `bench_opt_4bit.py:123-124` 搬运量 | 8-bit 只算一个状态（N） | ★ **算两个（2N）+ scale 每 64** |

#### 7.82.2 而验证是重跑，不是"改完了"

```
修之前:  配对比较 2.418 (离散 55.5%) ⇒ 【在噪声内 ✗ 不可判定】
修之后:  配对比较 2.568 (离散 24.5%)
         标准误 0.217 ⇒ ★【95% CI 2.142 ~ 2.994】⇒ 排除 1 ⇒ 【快 2.57×】✓

修之前:  8-bit 1.031 对 4-bit 1.031 ⇒ "省 1.00×"
修之后:  8-bit 2.125 对 4-bit 1.031 ⇒ ★【省 2.06×】✓

★ 而两次独立测量吻合:2.568（本次）对 2.627（7.80 那个修好的基准）
```

#### 7.82.3 ★ 而今天第五条"资产被错误描述"，指向一项具体的审计

| # | 资产 | 错在哪 | 错在哪【写下来的地方】 |
|---|---|---|---|
| ① | 循环 Transformer | 以为没接 | 无（是记忆问题） |
| ② | MaskGIT | 以为没有 | 无 |
| ③ | AVX2 4-bit 融合路径 | 以为缺实例 | 我们自己的判断（未查调用图） |
| **④** | **4-bit 优化器速度** | **"慢 4×"** | ★ **`adamw4bit.py` 的头注释** |
| **⑤** | **4-bit 优化器的 AVX2 路径** | **"没有"** | ★ **同一段头注释** |

**⇒ ④ 和 ⑤ 是【同一段文字里的两句】，而它们错了整整一轮 —— 而代价是【劝退使用者】。**

**⇒ 而 ① ② 是"不知道它存在"，③ 是"我推错了"，④⑤ 是【代码自己在说谎】。**
**⇒ 四类里只有第四类会【主动误导每一个读者，且持续误导**。

#### 7.82.4 所以该做的审计

**⇒ 把 bnb 源码里【所有关于性能的断言】捞出来，逐条对实测。**

```bash
# 候选模式
grep -rn "faster\|slower\|speedup\|[0-9]\+x \|x faster\|x slower\|no .* path\|falls back" \
     bitsandbytes/csrc/*.cpp bitsandbytes/bitsandbytes/**/*.py
```

**⇒ 而审计的判据（事先定）： **

```
一条性能断言【可信】当且仅当:
  ① 它写明了【对手】（比什么）—— "4x slower than the 8-bit AVX2 kernel" 写明了 ✓
  ② 它写明了【条件】（在哪测的）—— 而它没写 ✗
  ③ 它【有测量日期或引用】—— 而它没有 ✗

⇒ ★【④⑤ 那条满足 ① 却不满足 ②③ ⇒ 而"写明了对手"正是它显得可信的原因】
⇒ 这才是最危险的一类:【格式完整但数字过时】的断言
```

#### 7.82.5 而它的收益可以估

```
已知的两条错误断言 ⇒ 一条隐藏了 2.63× 的速度优势（且方向相反）
⇒ 而 bnb 的 csrc 里有【大量】这类"实测快 N 倍"的注释（本项目读过其中十几条）
⇒ ★【若错误率是 2/N,而每条错的都值一个数量级,那这项审计的期望值【高于写任何新内核】
```

**⇒ 而它还有一个本项目没享受过的好处：【不需要这台机器做什么】——
它是读代码 + 对照测量，而测量【大部分已经在本报告里了】。**

#### 7.82.6 所以下一步

**⇒ 做这项审计。方法：**

```
1. 把 csrc 和 Python 层里的性能断言全部列出（带行号）
2. 逐条标：对手 / 条件 / 日期 三项是否齐
3. 三项齐的先放行；缺条件或缺日期的【优先重测】
4. 而已经在本报告里测过的（GEMM 速率、attention、elementwise、分块阈值）直接对照
⇒ ★【产出是一张"哪些注释可信、哪些要改"的表】
```

**⇒ 而这条路的性质与今天所有别的路都不同：它不追求更快，它追求【别再自己劝退自己】。**

'''

m = re.search(r'^### 7\.82 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.82'
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
