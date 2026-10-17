"""Append 7.95: ③ could not be completed, and the reason is a mistake already on record.

The real-shape test failed, and it failed in a way worth recording because the project has hit it
before.
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

SEC = '''### 7.95 ★★ ③ 没能完成 —— 而失败方式是一个【已经记录在案】的错误

#### 7.95.1 发生了什么

**用 CLBlast 在真实形状（从两个探针实测得到）上重测 §7.89 的 1.81×。第一版跑出：**

```
gate_proj [6144,2048] M=128   单独基准 【46787.4 GFLOPS】
                              并发核显   【79438.3 GFLOPS】
⇒ 合计/CPU单独 = 【470.47×】
```

**★ 而 gfx902 的标称峰值是 【1150 GFLOPS】⇒ 那些数字是它的【40 倍】。**

#### 7.95.2 ★★★ 而这是本项目已经犯过一次的错

```
报告里早已记录:"iGPU 33446 GFLOPS 是把【入队时间】当成了【执行时间】"
⇒ ★【而我在 ③ 里又犯了一遍】
```

**⇒ 根因：我写的 `gpu_solo` 漏了 CLBlast 的返回码检查（原 `igpu_real_shapes.py` 里有
`if rc != 0: raise`）。CLBlast 报错后立即返回 ⇒ "耗时"只剩入队开销。**

**⇒ 而加了检查之后：**

```
RuntimeError: CLBlast 返回 -1011 —— 内核没跑
⇒ ★【所以那些 GFLOPS 是一个【从未执行的测量】的时间】
```

#### 7.95.3 ★★ 所以 ③ 的结论是【未完成】，而不是"不成立"

**CLBlast 在 M=128, N=6144, K=2048 这类形状上返回 **-1011**，而定位它需要的时间超出了本轮。
⇒ ★【所以"1.81× 在真实形状上成不成立"这个问题【仍然开着】。**

**⇒ 而这条比一个假数字重要：**

```
✗ 报告里若写进 470×,那是一条【彻底的假结论】,而它看起来像重大发现
✓ 而它被拦下的唯一原因是【数字超过了设备的物理上限】
⇒ ★【判据:任何超过设备标称峰值的核显数字,【先当测量错误处理,而不是当发现】】
```

#### 7.95.4 而三个教训叠在一起，指向同一件事

```
① §7.89 的 1.81× 是在【合成形状 + 手写内核】上测的 ⇒ 不能外推到真实形状(§7.68 已知它依赖 K)
② igpu_real_shapes.py 早已记录:这颗核显【形状依赖极强】—— 2048³ 是 CPU 的 3.21×,而 512³ 更慢
③ ★ 而本次连"能不能在这批形状上跑"都还没通过（CLBlast -1011）
⇒ ★【三步都指向:核显的可用性不是一个数,是一个【形状 × 内核 × 状态】的三维曲面,
   而每一次把它压成一个数,都得到过一个错的结果】
```

#### 7.95.5 所以今天这轮的账

| # | 任务 | 状态 |
|---|---|---|
| **①** | 真 LoRA 训练 + 出图对比 | ✅ **完成**：150 步 · 17.5 分钟 · 6.74 s/步 · 权重动了 · 输出变了 · **而质量下降，原因已知**（§7.94） |
| **②** | 换模型填分类表 | ✅ **完成**：bf16 LLM 96.5% 矩阵乘 · 无反量化；**而量化让更大的模型快 2.7×**（§7.93） |
| **③** | 真实形状上验 1.81× | ❌ **未完成**：CLBlast -1011，问题仍开着（§7.95） |

**⇒ 而 ③ 留了一条明确的下一步：先解决 -1011，它大概率是形状/内核选择的问题，
而 `igpu_real_shapes.py` 在 M=1024 的形状上是跑通的 ⇒ 从那个可用的配置往回退着试。**

'''

m = re.search(r'^### 7\.95 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.95'
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
