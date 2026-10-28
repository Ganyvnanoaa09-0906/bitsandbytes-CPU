"""Append 7.75: the 4-bit optimizer's real nature, and three traps that cost time tonight.

Preparing the "is it a tool or a toy" test turned up three things worth recording separately from the
result: what the optimizer actually trades, a Windows DLL loading trap its own docstring warns about
and I still walked into, and a housekeeping gap of my own making.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

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

SEC = '''### 7.75 4-bit 优化器的真实性质，以及今晚三个工具陷阱

#### 7.75.1 它换的是什么 —— 而它自己写在第一段

`bitsandbytes/bitsandbytes/optim/adamw4bit.py` 的头注释：

```python
# What it buys (measured, see DESIGN_4BIT_OPTIMIZER.md):
#     fp32 AdamW state   8.000 bytes/param
#     4-bit state        1.031 bytes/param    -> 7.8x less
#
# What it does NOT buy: speed. The kernel currently has no AVX2 path, so it runs
# about 4x slower than the 8-bit AVX2 kernel. The optimizer is 1-4% of a training
# step, so that is a few percent of wall clock -- acceptable for trying it out, and
# the AVX2 version is the next piece of work.
```

**⇒ 所以它换的是【内存 7.8×】，代价是【优化器内核慢约 4×】** —— 而优化器占一步 1~4%
⇒ 净代价是几步时钟的百分之几。

**★ 而"它是工具不是玩具"这件事的第一个证据就在这段注释里：**
**它第一段就说了代价。** 一个玩具会只写"省 7.8 倍"，让用户自己去发现那 4 倍。

#### 7.75.2 已验证的部分（跑它的测试套件）

```
✓ 内核与 numpy 参考【逐码一致】—— 数学/打包/absmax 全对（test_optimizer_4bit_cpu.py）
✓ 状态 1.031 字节/参数 对 fp32 Adam 的 8.000 ⇒ 省 7.8×
✓ 能训练:fp32 与 4-bit 两条 loss 曲线【都单调下降】（test_adamw4bit_train.py）
   而该测试的模型只有 427K 参数 ⇒ ★【这是小模型的证据,不是我们这个 30.5M 模型的】
```

**⇒ 所以"它能不能真训"这个问题，在【小模型】上已经答了；在【我们的模型】上还没有。**

#### 7.75.3 陷阱一：Windows 按【模块基名】去重 DLL

`adamw4bit.py` 的 `_find_dll` 注释里写着：

```
Windows 的 LoadLibrary 按【模块基名】去重 —— 一旦 libbitsandbytes_cpu.dll 这个基名
被某个进程内已加载的模块占用（例如先 import bitsandbytes 拉起的旧 site-packages 安装），
之后所有同名 CDLL 调用都会拿回那个旧模块，即使路径完全不同。
```

**⇒ 而我第一次写 A/B 脚本时把 `from measure_guard import ...` 放在了前面 ⇒ 它间接拉起了
bitsandbytes ⇒ 于是 `_find_dll` 拿到的是【装着的旧库（0.50.2.dev0）】，而那个不导出 4bit 符号
⇒ 报 `ImportError`，而错误信息说"找不到 DLL"，**而 DLL 就在那里。**

**⇒ 修法**：把加载 DLL 的动作放到【任何可能 import bitsandbytes 的语句之前】。改完就好。

#### 7.75.4 陷阱二：同一个模块有【两个副本】，而只有根目录那个能解析 DLL

```
D:\work\bnb-4bitopt\adamw4bit.py                                    ← 仓库根目录的副本
D:\work\bnb-4bitopt\bitsandbytes\bitsandbytes\optim\adamw4bit.py    ← 包内副本
```

**而 `_find_dll` 的候选路径是【相对自己所在目录】算的：**

```python
for rel in (os.path.join('bitsandbytes', 'bitsandbytes'),
            os.path.join('bitsandbytes'),
            ''):
    got = _try(os.path.join(here, rel, 'libbitsandbytes_cpu.dll'))
```

**⇒ 只有 `here` = 仓库根目录时，`here/bitsandbytes/bitsandbytes/libbitsandbytes_cpu.dll` 才存在。**

**⇒ 我一开始加载的是【包内】那个副本 ⇒ `here` 深了一层 ⇒ 三个候选全落空 ⇒ 同一个 ImportError。**

**★ 而通过的测试套件 import 的是根目录那个（`sys.path.insert(0, r'D:\\work\\bnb-4bitopt')`）
⇒ 所以这个坑不会在跑测试时暴露，只会在【按路径加载】时暴露。**

#### 7.75.5 陷阱三：我的清理弄坏了两个测试

**10-15 的 checkpoint 清理删掉了 `i5build/loop_ar3/ckpt.pt/ckpt_last.pt`，
而 `test_adamw4bit_real.py` 与 `test_4bit_memory_payoff.py` 都硬编码了它 ⇒ 两个测试挂掉。**

```
★ 那次清理【保留了报告引用过的每一个 checkpoint,并且逐个验证了存在】——
   而我【没有检查测试引用过哪些】。
⇒ ★【"报告引用过"是一个代理,而"会被用到"才是真正要保的东西】
```

**⇒ 而这和今天反复出现的那条是同一条：用一个【看起来相关】的判据去代替【真正关心的那个】。**

**⇒ 修法不是换成另一个路径（下次清理还会坏），而是改成候选列表 + 取第一个能加载的：**

```python
CK_CANDIDATES = [
    r'...\\ar32_run1\\ckpt_last.pt',   # 32x32 AR,30.5M,grid 32 —— 与原引用同构
    r'...\\ar_wd\\ckpt_last.pt',
    r'...\\ar_long\\ckpt_last.pt',
    ...
]
def _pick_ckpt():
    for p in CK_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(...)
```

**⇒ 两个测试已修好并通过。**

#### 7.75.6 而这一节的方法论价值

**三个陷阱都是【同一个东西在两处存在，而只有一处是对的】：**

| 陷阱 | 两份东西 | 哪份对 |
|---|---|---|
| DLL 基名去重 | 装着的库 vs 源码树的库 | **源码树的（先加载的那个）** |
| 模块两个副本 | 根目录 vs 包内 | **根目录（`here` 的深度对）** |
| 清理保留判据 | 报告引用过 vs 会被用到 | **会被用到的** |

**⇒ 而三者的共同点：****错的那个不会报错，它只是【静默地用了另一个】**。
**⇒ 这与 bnb 里那批"静默回退"是同一类问题，只是发生在工具链而不是内核里。**

'''

m = re.search(r'^### 7\.75 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.75'
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
