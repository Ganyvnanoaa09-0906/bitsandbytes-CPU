"""Append 7.76: the 4-bit optimizer is a tool. Real model, real data, verified.

The question was whether it is a tool or a toy. The answer required running it on the actual 30.5M
model with actual data, not on the small model its own test suite uses, and reading the result against
a criterion fixed in advance.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.76 ★ 4-bit 优化器是工具 —— 真模型、真数据、已验证

#### 7.76.1 A/B：30.5M 参数的本模型，真 token，同初始化同数据序

| 步 | 8bit loss | 4bit loss | 相对差 |
|---|---|---|---|
| 25 | 7.0490 | 7.0490 | −0.00% |
| 75 | 6.4860 | 6.4777 | −0.13% |
| 150 | 6.8410 | 6.8445 | +0.05% |
| **250** | 5.6876 | 5.7178 | **+0.53%（最差一格）** |
| 300 | 6.7351 | 6.7283 | **−0.10%** |

```
最终 loss:  8bit 6.7351 · 4bit 6.7283 ⇒ 相对差 【−0.10%】
步时:       8bit 5.5635 s · 4bit 5.3107 s ⇒ ★【0.955×（4bit 反而略快）】
状态内存:   8bit 66.3 MB · 4bit 33.4 MB ⇒ ★【省 2.0×】

判据（事先固定：最终 loss 差 <1% 且步时 <10%）⇒ ★【是工具】✓
```

#### 7.76.2 ★ 而两个数字必须如实讲清楚

**① 内存是 2.0×，不是 7.8×。**

```
头条那个 7.8× 比的是【4-bit 对 fp32 Adam(m+v)】
而 bnb 用户的实际对手是【AdamW8bit】:
   8bit  66.3 MB / 30.5M = 【2.17 字节/参数】
   4bit  33.4 MB / 30.5M = 【1.095 字节/参数】
⇒ ★【对 8-bit 是 2.0×】
⇒ 所以 7.8× 那个数【不错,但对手选错了】—— 在 bnb 的语境里,用户手里是 8-bit
```

**② 而"内核慢 4×"在真实训练里【完全看不到】。**

```
优化器占一步约 3% ⇒ 4× 的内核慢 = 3% × 4 = 【+0.27% 的步时】
而实测 0.955× ⇒ 噪声量级,而 4bit 还略快
⇒ 大概率是它【更小的状态内存流量】帮了缓存
```

#### 7.76.3 而它自己的两个测试，各有值得记的地方

**`test_adamw4bit_real.py`（真 30.5M 模型）：**

```
fp32 AdamW    232.7 MB (8.000 字节/参数)
4-bit          29.5 MB (1.014 字节/参数) ⇒ 省 7.9×
时间: fp32 645.15 s · 4-bit 619.73 s ⇒ 0.96×
```

**`test_4bit_memory_payoff.py` —— ★ 这是诚实报告的范本：**

```
✗ RSS 测量失败（全为 0）⇒ 【关于 batch 的结论不予给出】
   "可直接采信的只有上面的【优化器状态字节数】和【loss 轨迹】"
✓ 补充: 30.49M 参数 ⇒ 省的 203 MB 只占 15.4 GB 的 1.3%
     ⇒ ★【在这个规模上"换成更大 batch"的收益本就很小】✗
     收益随参数量线性放大（300M 参数约省 2 GB，占约 13%）✓
```

**⇒ 它自己宣布一个测量失败、划清可信范围、并主动说清收益在【小规模上很小**。

**⇒ 这正是"工具"的第二个标志：不替用户隐藏它的适用边界。**

#### 7.76.4 所以结论分三层

| 对手 | 内存 | 时间 | 适用 |
|---|---|---|---|
| **对 fp32 Adam** | **7.9×** | 0.96× | 从 fp32 迁过来的场景 |
| **对 AdamW8bit** | **2.0×** | 0.955× | ★ **bnb 用户的实际场景** |
| **在本模型（30.5M）上** | **省 203 MB = 内存的 1.3%** | — | ★ **收益很小** |
| **在 300M 参数上** | **约省 2 GB = 约 13%** | — | ★ **才是它的战场** |

**⇒ 所以"它是不是工具"的答案是【是】，而"它对我们有没有用"的答案是【在这个规模上几乎没有】——
因为 30.5M 参数的优化器状态本来只占 203 MB / 15.4 GB。**

**⇒ 而这两句话【不矛盾**，它们正是同一件工具的两个刻度。

#### 7.76.5 而这一节与前一条方法论的关系

**今天反复出现的错误是【用一个看起来相关的判据代替真正关心的那个】：**

```
✗ "报告引用过的 checkpoint" 代替 "测试会用到的 checkpoint"
✗ "7.8× 对 fp32" 代替 "对用户实际用的 8-bit"
✗ "内核快 4×" 代替 "训练步慢多少"
⇒ ★【而 4-bit 优化器这套东西,恰恰每一处都写清了【对谁、在什么规模下】】
```

**⇒ 所以它不止是工具 —— 它的文档方式本身就是本报告该学的。**

'''

m = re.search(r'^### 7\.76 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.76'
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
