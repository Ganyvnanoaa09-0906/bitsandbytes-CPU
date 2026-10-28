# -*- coding: utf-8 -*-
"""追加 §7.222.10：导出 ckpt 与 CPU ckpt 结构完全一致（可以直接当 resume 起点）。"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import io

TEXT = r"""

#### 7.222.10 导出的 ckpt 与 CPU ckpt 结构完全一致（可直接 resume）

M5 的验收不只是"能采样"，还要求导出的东西**能被 CPU trainer 当 resume 起点**。逐项对比
`ar_run_full3/ckpt_12000.pt`（CPU）与 `ar_run_igpu/ckpt_14100.pt`（iGPU 续训 100 步后导出）：

| 项 | CPU ckpt | iGPU ckpt | 判定 |
|---|---|---|---|
| 顶层键 | `cfg, cluster_of, model, offset_of, step` | 同 | ✅ |
| `state_dict` 键数 | 125 | 125 | ✅ 集合完全相同 |
| 张量形状 | — | — | ✅ 全部一致 |
| dtype | float32 | float32 | ✅ |
| `cfg` 逐字段 | — | — | ✅ **零差异** |

所以 iGPU 训出来的 ckpt 是**可以直接替换** CPU 的 ckpt 的：`train_ar_v2.py --resume`、
`sample_ar.py --ar`、`is_cpu_eval` 这类下游工具一行都不用改。这一点由
`ar_v2_export_test.py` 的 `load_state_dict(strict=True)` + 前向 loss 差 7.7e-5 一起证过。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))
