"""Append 7.84: the audit's result, narrowed, and a correction to my own reading within it.

Working through the candidates found exactly one genuine contradiction between comment and code, and
two of my initial suspicions were wrong. Recording both, because the narrowed result is more useful
than the alarming one.
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

SEC = '''### 7.84 审计结果收窄：只有一处真矛盾，而我自己的两处怀疑是错的

#### 7.84.1 审计方法与候选

```
① 捞所有性能断言（csrc 里 62 处）
② 捞所有"关闭/已撤/暂时/不启用"类断言（7 处）
③ 逐条对代码 —— 因为"声称关闭"是最容易被代码推翻的一类
```

#### 7.84.2 ★ 唯一真正的矛盾：L3738-3748

```cpp
L3738-3747 （注释）: "AVX2 路径【暂时关闭】… 打开方式：把 false 改成 (two_state && has_avx2_cpu())"
L3748 （代码）:      use_avx2 = (two_state && has_avx2_cpu());     ← 已经打开了
L3754 （代码）:      if (use_avx2) { optimizer_4bit_blockwise_avx2<float>(...); break; }
```

**⇒ ✓ 确认为真矛盾。** 而 L3742 那句"打开方式：…"证明有人改了却没删说明。

#### 7.84.3 ★ 而我的两处怀疑【是错的】，这里更正

**怀疑一：L498 说"NT 不启用"，而 L603 有 NT store ⇒ 以为矛盾。**

```
L498 （在 avx2_dequant_4BIT 里）:
    // NT store 分档：2026-09-22 实测【不启用】
    // 原因：本 kernel 只跑 ~5.5 GB/s，瓶颈是【LUT 位平面 shuffle 解码】（指令受限），
    //       不是存储通路 ⇒ NT 无从发挥。

L556 （在 avx2_dequant_8BIT 里，是另一个函数）:
    // 非临时存储（NT store）分档，2026-09-22
    //     输出 <= 1.0 MB : 普通存储赢（NT 亏 0.69~0.73x）
    //     输出 >= 4.2 MB : NT 赢 2.0~3.1x
    const bool use_nt = (sizeof(T) == 4) && bnb_is_aligned_for_nt(out) && …;
```

**⇒ ★【两条都是对的，它们说的是【两个不同的函数、不同的瓶颈】。**
**⇒ 而 L603 的 NT store 在 dequant_8BIT 里，不在 L498 说的那个函数里。**

**怀疑二：L3622"NT 已撤销" ⇒ 以为和代码矛盾。**

```
L3622: // ⚠️ 2026-09-22：NT store 在此 kernel 上实测【倒退 21%】，已撤销。
L3630: (void)use_nt_p;
⇒ ✓【变量留着但被 (void) 掉 ⇒ 与注释一致】。而且它解释了原因
   （读-改-写模式下 cache line 已在缓存中，强制刷出只会增加流量）。
```

**⇒ 两处怀疑都不成立。而我是在【读了半句就下判断】之后才核对的。**

#### 7.84.4 所以审计的真实结果

| 候选 | 判定 |
|---|---|
| L3738 "AVX2 暂时关闭" | ✗ ★ **真矛盾（代码已打开）** |
| L498 "NT 不启用" | ✓ **对**（那是 4-bit 反量化，确实不该用 NT） |
| L2248 "行复用已撤" | ✓ **对**（L2278 走融合路径） |
| L3622 "NT 已撤销" | ✓ **对**（`(void)use_nt_p`） |
| 其余 5 处性能断言 | ✓ **带日期/条件/数字，抽样核对无矛盾** |

**⇒ ★【所以真正的错误率是 1/7，而不是我预感的"大量"】。**

#### 7.84.5 ★ 而这 1/7 的分布才是关键

```
那 6 处【对】的注释,有一个共同点:
  · 它们都写明了【为什么】（"瓶颈是 LUT 解码，不是存储通路"）
  · 或者写明了【条件】（"输出 ≥ 4.2 MB"）
  · 或者留了【可核对的痕迹】（(void)use_nt_p）

而那 1 处【错】的:
  · 它只说了【状态】（"暂时关闭"）和【怎么做】（"把 false 改成…"）
  · ⇒ ★【它没有一个能自我证伪的成分】
```

**⇒ ★【所以修法不是"定期检查所有注释"，而是【让每条状态类断言带一个可核对的痕迹】——
比如引用设置它的那一行，或引用一个能跑出结论的测试名。】

**⇒ 而 L3738 那个错误如果写成"打开方式：见 L3748"，它就会在 L3748 被改时被一起看到。**

#### 7.84.6 而这一条让 7.83 的结论要收敛

**7.83 说"注释和代码在打架"—— 而审计表明【只有一处】在打架，
其余六处是【历史记录，且都标清了为什么】。**

**⇒ 所以正确表述是：**

```
✗ 过头的: "这个文件的注释不可信"
✓ 准确的: ★【"状态类"注释（关闭/启用/已撤）会随代码演进失效,
             而"原因类"注释（为什么不用）不会 —— 因为原因不随代码变】
```

**⇒ 而 adamw4bit.py 读到的恰好是一处状态类注释（"暂时关闭"），
⇒ ★【它读的那个来源,正好是七处里唯一失效的那一处 —— 而这不是运气,
   是因为它【只读状态类的话】: 头注释要的是"现状",而现状正是状态类注释最不可靠的地方。】**

#### 7.84.7 而审计的最终产出

```
① 一处必改:cpu_ops.cpp L3738-3747 的"暂时关闭"（与 L3748 矛盾）
② 一条约定:状态类断言必须带可核对痕迹（引用行号或测试名）
③ 一条判据:原因类注释可信,状态类注释必须核对代码
④ ★ 而 7.82 的四条判据里,第四条（现状 vs 历史）要改成:
     ★【原因 vs 状态】—— 前者不失效,后者必然失效
```

**⇒ 而这条比"哪个数字过时"有用得多：它给出了【哪些注释可以信、哪些必须查】的规则。**

'''

m = re.search(r'^### 7\.84 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.84'
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
