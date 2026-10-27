# -*- coding: utf-8 -*-
"""追加 §7.222.12（独占 60 步的干净吞吐）+ 更新 plan 的 M6 行。"""
import io

TEXT = r"""

#### 7.222.12 核显独占：连续 60 步 **2.106 s/step = 972 tok/s**（比 CPU 快 1.71×）

在 CPU trainer 挂起的状态下（`_t_fair_train.py`，`finally` 保证恢复）从 `ckpt_16000` 续训 60 步：

| 项 | 值 |
|---|---|
| 步时 | **median 2.106 s/step**（min 2.061，第一拍 8.168s 是 CLBlast 首次调优） |
| 吞吐 | **972 tok/s** |
| 对照 CPU | 3.1~4.0 s/step = 509~569 tok/s ⇒ **核显快 1.71×** |
| loss | 3.7538 → 5.7642（单 batch 快照，mean 5.2435，与 CPU 同期分布一致） |
| gnorm | 0.71~1.59（clip 1.0 生效） |
| 导出 | `F:\weibo_cos\ar_run_igpu\ckpt_excl.pt` |
| CPU 恢复 | rc=0，进程仍在 458%，日志继续 ✅ |

把 §7.222.11 的对照表和这一条合起来看，最终口径是：

| 场景 | 一步 | 吞吐 |
|---|---|---|
| CPU 单跑（6 线程） | 3.1~4.0 s | 509~569 tok/s |
| **核显单跑（独占）** | **2.11 s** | **972 tok/s**（**1.71×**） |
| 核显 + CPU 同时跑 | 4.05 s | 505 tok/s（两者互相拖累） |

**这就是这个移植最该被记住的数字**：核显不是"更快一点"，而是**在独占条件下快 1.7 倍**；
之前所有"更慢"的测量都是并发污染。使用建议也随之明确：**核显训练要把 CPU 让出来**，
或者至少把 CPU 侧线程数压到 1~2（带宽争抢是按字节计，不是按核计）。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace(
    """**但比 CPU 的 3.1s/step（569 tok/s）慢 30~40%** ⇒ 要真用起来必须先优化 attention（见下）。""",
    """**⚠️ 更正（见报告 §7.222.11/12）：之前的 4.0~4.4s 是在 CPU trainer 满载 460% 时测的，被共享 DDR4 带宽/功耗污染。
用 NtSuspendProcess 挂起 CPU 后重测：median 2.106 s/step = 972 tok/s，比 CPU 的 569 tok/s 快 1.71×。**""")
s = s.replace(
    """**吞吐结论**：device-only 4.0s/step vs CPU 3.1s/step。""",
    """**吞吐结论（更正版）**：核显独占 **2.11 s/step = 972 tok/s**；CPU 569 tok/s；两者并发时核显掉到 4.05s/step。
并发污染这个坑足够大，单独写了 §7.222.11。""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('appended 60-step section + plan updated')
