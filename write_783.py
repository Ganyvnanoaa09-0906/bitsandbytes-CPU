"""Append 7.83: the audit's first finding, and it is deeper than a stale number.

The comments and the code in cpu_ops.cpp disagree about whether the 4-bit AVX2 path is even enabled.
The comments say it is off. The code turns it on and dispatches to it. Reading only the comments is
exactly what produced the two false sentences in adamw4bit.py.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.83 ★ 审计的第一个发现：注释和代码在打架

#### 7.83.1 cpu_ops.cpp 里的三层互相矛盾

```cpp
L3738-3747 （注释）:
    // ★ AVX2 路径【暂时关闭】：实测它产出错误的码
    //   s1[:8] = [135 112 128 0 137 144 144 0]，奇数下标全是 0 ✗
    //   ⇒ nibble 打包有 SIMD 布局 bug（数学部分经逐码测试是对的一版 ✓）
    //   ⇒ 先回退到已验证的标量路径，代码留着供后续定位 ✓
    //   ★ 打开方式：把 false 改成 (two_state && has_avx2_cpu())

L3748 （代码）:
    use_avx2 = (two_state && has_avx2_cpu());       ← ★【它已经打开了】

L3754:
    if (use_avx2) { optimizer_4bit_blockwise_avx2<float>(...); break; }   ← ★【而且真的走】
```

**⇒ ★【注释说关闭，代码是打开的，而分派真的走 AVX2。】**

**⇒ 而 L3742 那句"打开方式：把 false 改成 …"说明：有人【改了那个 false】，
但【没删掉那句说明】。**

#### 7.83.2 而 7.80 那个"慢 4.6 倍"也有了出处

```cpp
L2923-2927 （在 optimizer_4bit_blockwise_scalar 里）:
    // 【成对处理】—— 性能关键。
    // 逐元素 get/set 会对同一个字节做"读-改-写"，相邻两个元素之间产生
    // store-to-load forwarding 停顿，而且彻底阻断向量化：实测比 8-bit 版
    // 慢 4.6 倍（7.385 ms vs 1.591 ms，1M 参数）。
    // ★ 改成一次读字节、拆两个码、算完再打包写一次之后，每个字节只碰一次。

L2931-2941  就是【改成之后】的代码（一次读字节 / 拆两个码 / 打包写一次）
```

**⇒ ★【所以"慢 4.6×"是【修复前】的数字，而那一段是【修复记录】。】**

#### 7.83.3 ★ 所以 adamw4bit.py 的错误有两层

```
① 它把一份【修复记录】读成了【现状描述】
   —— 而那段注释自己写了"改成…之后"，只是那个数字在后面
② 而它读的是【标量函数】里的注释
   —— 而 AdamW4bit 走的是【AVX2 路径】（L3748 已打开）
⇒ ★★【数字是真的、函数是真的、修复也是真的 —— 而【读错了那一个、也读错了时间】
```

#### 7.83.4 ★★ 而这才是今天最深的发现

**⇒ `adamw4bit.py` 的错【不是抄错数字】，而是【它读的那个来源，本身就和代码不一致】。**

```
源码里的注释:  一致于【更早的一个状态】
源码里的代码:  已经前进
⇒ ★【只读注释的人,得到一个【曾经正确】的图景】
```

**⇒ 而这不是这个文件的问题，是一个【注释与代码同时演进但不保证同步】的必然结果：**

| 变动 | 代码 | 注释 |
|---|---|---|
| 修 nibble 打包 | ✓ 改了 | ✗ "暂时关闭" 没删 |
| 打开 AVX2 | ✓ 改了 | ✗ "打开方式：…" 没删 |
| 标量路径成对处理 | ✓ 改了 | ✓ **但"慢 4.6×"留着**（作为修复记录，那是合理的） |
| **Python 头注释** | — | ✗ **读了上面两处，得到完全错误的结论** |

**⇒ 而第三行是【合理的】—— 修复记录就该留（它解释了为什么代码长这样）。
⇒ ★【问题在于没有一个约定，区分"这条注释描述现状"和"这条注释记录历史"。】**

#### 7.83.5 所以审计的判据要加一条

**7.82.4 定的三条（对手 / 条件 / 日期）不够，要加第四条：**

```
④ ★【这条断言描述的是【现状】还是【历史】?】
   · 描述现状 ⇒ 必须与【当前代码】一致
   · 记录历史 ⇒ 必须【显著标注】（如"修复前"、"已撤"、"2026-09-22 之前"）

⇒ 而 L2925 那种写法（"实测慢 4.6 倍" 紧跟 "改成…之后"）是最危险的:
   它在【同一段】里同时描述历史和现状,而读者可能只取前半句
⇒ ★【正确写法:"修复前实测慢 4.6×（7.385 vs 1.591 ms）；修复后见 L2829 的分档数据"】
```

#### 7.83.6 ★ 所以这轮审计的预期产出变了

**⇒ 原本以为产出是"一张哪些数字过时的表"。而第一个发现说明产出应该是：**

```
① 一张【注释与代码不一致】的清单（比"数字过时"更根本）
② 一条约定:性能断言必须标明【现状 vs 历史】
③ 而对 adamw4bit.py 那种【外部读者】,正确做法是【读代码,不读注释】
   —— 或至少【两者都读,不一致时以代码为准】
```

**⇒ 而这一条与今天那条主教训是同一件事的更深一层：**

```
浅层: 别用代理代替真实测量     ⇒ 用注释代替代码
深层: ★【而"注释"看起来比"代码"更像一个答案 —— 它是给人读的,所以更像答案】★★
```

'''

m = re.search(r'^### 7\.83 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.83'
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
