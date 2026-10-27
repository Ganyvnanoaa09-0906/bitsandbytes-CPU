"""Report 7.125: 500-step real-model iGPU run completed + Linux readiness (WSL findings).

Numbers:
  * deepseek-coder-1.3B LoRA, seq=64, 500 steps: 7.392 s/step, loss 11.7175 -> 0.0000 (synthetic fixed
    batch overfit), saved trainable.pt 29.7 MB; parity 1.91e-6.
  * GPU utilisation sampled during a 30-step run (76 samples @1.5 s): max 125 %, average 59.4 %,
    60/76 samples >= 50 %, 11/76 < 20 % (the CPU-only phases between iGPU calls); CPU average 90.4 %.
    This answers "为什么任务是核显但 CPU 也满" -- the per-op hybrid interleaves CPU and iGPU work.
  * Linux: WSL Ubuntu has no /dev/dri, no libOpenCL, no torch -> iGPU OpenCL training needs a real Linux
    install (Mesa Clover/Rusticl + libclblast). bnb CPU backend works in WSL (verified builds/selftests).
    Scripts made portable via CLBLAST_PATH; README_LINUX.md lists the checklist and the honest caveat
    that all performance numbers come from the Windows driver and must not be assumed to carry over.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW = '''### 7.125 真实模型 500 步核显训练完成；Linux 侧现状与清单

**① deepseek-coder-1.3B（LoRA，seq=64）500 步跑完**：

| 项 | 结果 |
|---|---|
| 步时间 | **7.392 s/步**（中位数；iGPU GEMM 3.0–4.0 s） |
| loss | 11.7175 → **0.0000**（固定批过拟合，验证学习能力） |
| CPU/iGPU parity | **1.91e-6** |
| 权重 | `trainable.pt` 29.7 MB 已保存 |
| 显存/内存 | 训练后宿主空闲 9.7 GB（LoRA 释放 base CPU 副本后） |

**② "核显在干活吗"的量化答案**：30 步运行期间每 1.5 s 采一次 GPU 引擎利用率（76 个样本）：

| 指标 | 数值 |
|---|---|
| GPU 利用率均值 | **59.4 %** |
| 峰值 | **125 %**（多引擎求和） |
| ≥50 % 的样本 | 60 / 76 |
| <20 % 的样本 | 11 / 76（CPU 段） |
| 同期 CPU 平均 | 90.4 % |

⇒ 逐算子混合的形态就是"CPU 与核显交替"，所以 CPU 满、GPU 半载是设计使然，不是没用核显。要
"CPU 闲、GPU 满载"只有设备常驻引擎能做到（§7.122，未完成）。

**③ Linux 侧盘点（WSL 实测）**：`/dev/dri` 不存在、`libOpenCL*` 缺失、torch 未装——WSL2 的 GPU
直通面向 D3D12/CUDA，**不提供 iGPU 的 OpenCL**。因此：

- Linux 上跑核显训练需要**真 Linux**：Mesa OpenCL（Vega 6 走 Clover；Rusticl 对新卡更好）+
  CLBlast + pyopencl，`clinfo` 能看到 gfx902；
- WSL 能做的：**bnb CPU 后端**（`build_linux.sh` 已验证）与 CPU 训练基线；
- 训练脚本已做可移植：`CLBLAST_PATH` 环境变量，Linux 默认 `libclblast.so`；
- **诚实边界**：所有性能数字来自 Windows + AMD 官方 OpenCL 驱动，Linux Mesa 上不能假设复现；
  换平台先跑 `--check` 的 CPU/iGPU 对拍；
- 清单见仓库 `gt1030/README_LINUX.md`。

'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW + '# 附录 A：AR 训练路径的实测证据', 'insert 7.125'),
    ('**本文是 124 节的【时序记录】', '**本文是 125 节的【时序记录】', 'header: 125'),
    ('**本报告的 124 节里有 12 节', '**本报告的 125 节里有 12 节', 'F header: 125'),
    ('| **通用 iGPU 训练套件（任意模型）** | ★ **17 架构 ×2 模式 34/34 通过，CPU/iGPU parity 全 0.0；真实 1.3B parity 1.9e-6；--data/--save/自动下载** | §7.124 · `gt1030/igpu_train.py` |',
     '| **通用 iGPU 训练套件（任意模型）** | ★ **22 架构 ×2 模式 44/44 通过，parity ≤4.8e-7；真实 1.3B 500 步 7.39 s/步、权重已存** | §7.124–7.125 · `gt1030/igpu_train.py` |',
     'H table: 7.125 row'),
]


def main() -> int:
    text = io.open(REPORT, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        if out.count(old) != 1:
            print('ABORT %-20s anchor=%d' % (label, out.count(old)))
            return 1
        out = out.replace(old, new, 1)
        print('ok    %s' % label)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(out)
    print('report chars %d -> %d, lines %d -> %d'
          % (len(text), len(out), text.count('\n') + 1, out.count('\n') + 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
