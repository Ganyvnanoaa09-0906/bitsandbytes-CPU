# -*- coding: utf-8 -*-
"""追加 §7.222.16：优化后 80 步持续运行结果（median 1.350 s/step = 1517 tok/s，2.67× CPU）。"""
import io

TEXT = r"""

#### 7.222.16 优化后的持续运行：**80 步 median 1.350 s/step = 1517 tok/s（2.67× CPU）**

前面 §7.222.13~15 的数字是 4~5 步的中位数。为了确认不是"短跑虚高"，在 CPU trainer 挂起的
独占窗口里跑了 **80 步**（`_t_fair_train.py --steps 80`，从 `ckpt_16000` 起、lr 5e-6、尾部 2% 做 val 不参与训练）：

| 项 | 值 |
|---|---|
| 步时 | **median 1.350 s/step**（min 1.329，第一拍 7.284s 是 CLBlast 首次调优） |
| 吞吐 | **1517 tok/s** |
| 对照 CPU | 3.1~4.0 s/step = 509~569 tok/s ⇒ **核显快 2.67×** |
| 稳态性 | 80 步里步时 1.33~1.43s，没有漂移、没有驱动故障 |
| loss | 3.7537 → 6.0211（单 batch 快照，mean 5.3326，与 CPU 同期分布一致） |
| 导出 | `F:\weibo_cos\ar_run_igpu\ckpt_excl80.pt`（CPU 同格式，可 resume/采样） |

**这个数字可以当结论用**：30M AR 模型在 gfx902 核显上 **1.35 s/step / 1517 tok/s**，
是同期 CPU（6 线程）的 **2.67 倍**，且数值语义与 torch fp32 一致（10 步轨迹 1.98e-3）。

**剩下没做完的优化（都是 2~5% 量级，已记录的盘子）**：
- 层间 x 传输还走 host 往返（每层 3.4ms，前后向共 ~68ms/step）——同 context 内本可以 device→device
- 20 多个小 kernel（rms/split/silu/add/scale）合计 ~280ms/step
- device AdamW 40ms（40 次 launch）
理论下限约 1.1~1.2 s/step（≈3× CPU）。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace("**吞吐结论（更正+优化后）**：核显独占 **1.338 s/step = 1531 tok/s**",
              "**吞吐结论（更正+优化后，80 步持续测）**：核显独占 **1.350 s/step = 1517 tok/s**")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')
