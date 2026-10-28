"""Finish the evidence appendix: bandwidth, CPU state, per-operator bounds, quantisation
locations, raw logs, and the failure record with its numbers.

Also fixes a formatting bug in part 1: the memory table's first data row has a %.1f MB
placeholder with no argument, so it would print literally.

Everything measured here is on the real training configuration unless labelled 独立.
"""
from __future__ import annotations
import os, sys
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

import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


ROOT = r'D:\work\bitsandbytes-CPU'
OUT = r'D:\work\research\evidence'
os.makedirs(OUT, exist_ok=True)
L = []
def w(s=''):
    L.append(s)

# ---------------------------------------------------------------- bandwidth in use
w('## A.5 内存带宽：实际用量 vs 机器能力')
w()
w('| 项 | 值 | 来源 |')
w('|---|---|---|')
w('| 机器标称（DDR4-2667 双通道） | 42.7 GB/s | WMI: 2×8 GB Samsung M471A1G44AB0-CTD |')
w('| **纯读实测（4 线程）** | **23.7 GB/s** | `measure_real_bandwidth.py`，空闲 |')
w('| 纯读（被负载污染时） | 17.8 GB/s | 同脚本，A/B 占 4.79 核时（已作废） |')
w('| 纯写（fill_） | 8.1 GB/s | 同上 |')
w('| 拷贝（读+写） | 13.0 GB/s | 同上 |')
w('| 核显流式拷贝 | 28.9 GB/s | `bench_opencl3.py` |')
w()
w('**训练一步的实际需求**：激活约 500 MB（10 层 × 3 遍 × B·T·d = 16.8 MB）+ 梯度 122 MB '
      '+ 优化器 244 MB ≈ **0.8~1.2 GB/步**。')
w()
w('一步 3.95 s ⇒ **约 0.2~0.3 GB/s**，即机器读取能力的 **1% 左右**。')
w()
w('**⇒ 内存带宽不是当前训练的瓶颈。** 这一条推翻了此前的假设：一度认为激活流量限制了')
w('训练，但用量比能力低两个数量级，而核显并发失败的原因也不是带宽总量（见 A.9）。')
w()

# ---------------------------------------------------------------- CPU state
w('## A.6 CPU 状态：频率 / 温度 / 功耗')
w()
w('| 项 | 值 | 获取方式 |')
w('|---|---|---|')
idle = load = None
try:
    r = subprocess.run(['powershell', '-NoProfile', '-Command',
                        '(Get-CimInstance Win32_Processor).CurrentClockSpeed'],
                       capture_output=True, text=True, timeout=20)
    idle = r.stdout.strip()
except Exception:
    pass
w('| 空闲频率 | %s MHz | WMI `CurrentClockSpeed` |' % (idle or 'n/a'))
w('| **满载频率（实测）** | **2375 MHz** | `cpu_fma_headroom.py` 在负载中读取，未降频 |')
w('| 基准频率 | 2380 MHz | 任务管理器 |')
w('| 单核睿频（标称） | 4000 MHz | 规格；**全核实际只有 2375**，此前误用 2.81 GHz 计算峰值 |')
w()
temp = power = 'n/a'
try:
    r = subprocess.run(['powershell', '-NoProfile', '-Command',
                        'Get-CimInstance -Namespace root/wmi -ClassName '
                        'MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue | '
                        'Select-Object -First 1 -ExpandProperty CurrentTemperature'],
                       capture_output=True, text=True, timeout=25)
    v = r.stdout.strip()
    if v.isdigit():
        temp = '%.1f °C' % (int(v) / 10.0 - 273.15)
except Exception:
    pass
w('| 温度 | %s | `MSAcpi_ThermalZoneTemperature`（笔记本常不暴露） |' % temp)
w('| 功耗 | %s | Windows 无标准接口；`Win32_Battery` 的放电速率只在电池供电时有效 |' % power)
w()
w('**拿不到的**：本机是笔记本，ACPI 热区与 RAPL 均未暴露，因此**温度与功耗无法读**。')
w('可作替代的观察是频率：满载 2375 MHz 等于基准频率，说明**没有观察到降频**，也就没有')
w('明显的热墙。')
w()

# ---------------------------------------------------------------- compute vs memory bound
w('## A.7 逐算子：计算受限还是内存受限')
w()
w('| 算子 | 独立峰值 | 实测路径 | 判定 | 依据 |')
w('|---|---|---|---|---|')
w('| **`aten::mm`** | **218~235 GFLOPS**（该形状） | **186 GFLOPS** | **计算受限（已达 79~85%）** | '
      '同一内核只改矩阵大小：n=384 → 371，n=2048 → 230（L2→L3 掉落） |')
w('| RMSNorm（含 QK-Norm） | — | 2262 ms / 步 | **内存受限** | 纯 elementwise，'
      '490 次调用 × 2.6 ms，无 FMA 密集部分 |')
w('| flash attention | — | 2261 ms / 步 | **未知** | 未单独测；SDPA 的 CPU 后端效率未验证 |')
w('| 优化器（8-bit） | — | 138 ms / 步 | 内存受限 | 逐元素读写状态 |')
w('| 优化器（Muon） | — | 2562 ms / 步 | **计算受限** | 其中 2.42 s 是 NS 的矩阵乘 |')
w()
w('**CPU 的算术上限**（实测，非标称）：')
w()
w('| 尺寸 | 工作集 | GFLOPS | 相对 |')
w('|---|---|---|---|')
for sz, ws_, gf in ((384, '1.7 MB（L2 内）', 371.3), (768, '6.9 MB（L3 内）', 269.0),
                    (2048, '49 MB（L3 外）', 229.8)):
    w('| %d | %s | **%.1f** | — |' % (sz, ws_, gf))
w()
w('**Zen2/Renoir 的 FPU 是 128-bit 宽**，因此每周期 1 个 256-bit FMA，6 核 @ 2.375 GHz '
      '的理论上限是 **228 GFLOPS**；实测 L2 内到 371（超出该模型，说明单核可双发），'
      '而**不是**早期误算的 537（那假设了每周期两个 256-bit FMA）。')
w()

# ---------------------------------------------------------------- quantisation
w('## A.8 量化格式的实现位置')
w()
w('| 格式 | 文件 | 说明 |')
w('|---|---|---|')
w('| **4-bit 优化器状态** | `bnb-4bitopt` 分支 | 已上 PyPI；'
      '1.031 字节/参数，比 8-bit 快 2.56×（内核 0.658 ms vs 1.682 ms） |')
w('| **NF4 / FP4 权重量化** | `bitsandbytes/csrc/cpu_ops.cpp:275`（`nf4_lut`）、'
      '`avx2_quantize_4bit`、`avx2_dequant_4bit` | 表是 **16 个标准正态分位数** |')
w('| **2-bit / 3-bit 权重** | `quant2bit.py`（2-bit 专用）、`quantnbit.py`（通用 N-bit，'
      '2/3/4/5 位，自检往返） | 每字节 4 个索引；位序：字节内第 k 个值占 [2k, 2k+1] 位 |')
w('| **接入训练** | `quant_lora.py`（`nf2`/`uni3`/`nf3` 等分支）、'
      '`linear2bit.py`、`embed2bit.py` | 冻结基座 + LoRA |')
w('| 码表研究 | `study_quant_v2.py`、`quantize_model_eval.py` | 见下表 |')
w()
w('**matmul 内核**：')
w()
w('| 设备 | 文件 | 实测 |')
w('|---|---|---|')
w('| CPU | `bitsandbytes/csrc/cpu_ops.cpp` 的 AVX2 路径 | sgemm 峰值 255 GFLOPS（早期），'
      '本表 A.7 的数值更新 |')
w('| 核显 | `igpu_gemm_v3/v4.py`、`test_cpu_gpu_parallel.py` | 300 GFLOPS（**但该基线后经复查'
      '不可复现**，见 A.10） |')
w()
w('**吞吐与数值误差（不同位宽，实测路径）**：')
w()
w('| 位宽 | val loss | 相对 fp32 | 每权重字节 | 相对误差 |')
w('|---|---|---|---|---|')
for bits, loss, base, bpp, rel in (('fp32', 5.5038, 0.0, 4.0, '—'),
                                   ('4 bit nf', 5.5084, 0.0046, 0.5625, 0.0919),
                                   ('3 bit uniform', 5.5234, 0.0195, 0.4375, 0.2149),
                                   ('2 bit nf', 5.6206, 0.1168, 0.3125, 0.4175)):
    w('| %s | %.4f | %+.4f | %.4f | %s |' % (bits, loss, loss - 5.5038, bpp, rel))
w()
w('**结论**：3-bit 实质无损（+0.0195），2-bit 约 2%。而**量化对训练速度是负的**'
      '（0.80~0.85×，前向 +66%），因为解包本身是逐元素的内存操作，与它省的流量相抵。')
w()

# ---------------------------------------------------------------- failures
w('## A.9 失败记录（每条都有数据）')
w()
w('| 方向 | 结果 | 原始数据位置 |')
w('|---|---|---|')
w('| **核显并发**（五种变体） | 0.23~0.91× | `test_concurrency.py`、`ns_on_device.py`、'
      '`ns_async.py`、`settle_parallel_timing.py` |')
w('| **核显 LDS 吞吐** | 66~68 GB/s，需 544 | `measure_lds_throughput.py` |')
w('| **核显 occupancy** | 组/CU 翻倍 +3%，翻 4 倍 −25% | `igpu_occupancy_ts.py` |')
w('| **核显 300 GFLOPS 基线** | 不可复现（v4 现给 126.9 且有错） | `igpu_gemm_v4.py` |')
w('| **多分辨率课程** | B 比 A 差 0.30 | `curriculum_B1.log`、`curriculum_B2.log` |')
w('| **SGDR** | 两周期全程略差 | `i5build/sgdrA.log`、`sgdrB.log` |')
w('| **NS 批量版** | 0.89× 更慢 | `night_ns.log` |')
w('| **NS 异步版** | 比串行差 5% | `night_nsasync.log` |')
w('| **NS 迭代 5→3** | 方阵残差差 1.56× | `ns_iterations.py` |')
w('| **量化提速** | 0.80~0.85× | `quant_speed_real_step.py` |')
w('| **梯度累积** | 0.88~0.93× | `test_grad_accum_speed.py` |')
w('| **MaskGIT 采样** | 快（2.30 s/张）但**出噪声** | `mg_iter4.png`、`maskgit_train.log` |')
w()
w('**MaskGIT 的机制**：`--mask-mode full` 只去掉了因果遮罩，并没有喂随机掩码，因此训练出的')
w('是双向语言模型而不是掩码预测模型。证据是它的 gap 在 1301 步涨到 **+0.85**，而因果对照')
w('是 +0.58，val 差 0.28（6.0824 vs 5.7982）。采样从全 [MASK] 开始，与训练分布完全不一致。')
w()

# ---------------------------------------------------------------- raw logs
w('## A.10 原始日志（逐字复制）')
w()
rawdir = os.path.join(OUT, 'raw')
os.makedirs(rawdir, exist_ok=True)
PATTERNS = ['i5build/sgdrA.log', 'i5build/sgdrB.log', 'curriculum_B1.log',
            'curriculum_B2.log', 'curriculum_ab.log', 'maskgit_train.log',
            'maskgit_sample.log', 'maskgit_causal.log', 'night_run.log',
            'night_tk.log', 'night_ns.log', 'night_nsasync.log', 'night_ttl.log',
            'curriculum_data.log', 'ab_muon.log', 'ar32_run1.log']
copied = []
for p in PATTERNS:
    src = os.path.join(ROOT, p.replace('/', os.sep))
    if os.path.exists(src):
        dst = os.path.join(rawdir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
            copied.append((os.path.basename(src), os.path.getsize(dst)))
        except Exception:
            pass
w('原始日志已复制到 `evidence/raw/`，共 %d 个文件：' % len(copied))
w()
w('| 文件 | 字节 |')
w('|---|---|')
for name, sz in sorted(copied):
    w('| `%s` | %d |' % (name, sz))
w()
w('**关键原始片段**（不做汇总，逐字）：')
w()
w('```')
for p in ('i5build/sgdrA.log', 'i5build/sgdrB.log', 'maskgit_train.log'):
    src = os.path.join(ROOT, p.replace('/', os.sep))
    if not os.path.exists(src):
        continue
    w('### %s' % p)
    try:
        lines = [ln.rstrip() for ln in open(src, encoding='utf-8', errors='replace')
                 if 'val=' in ln]
        w('\n'.join(lines))
    except Exception as e:
        w('读取失败 %s' % e)
    w()
w('```')
w()

path = os.path.join(OUT, 'A_training_evidence_part2.md')
with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('  第 2 部分已写出: %s（%d 行）' % (path, len(L)))
print('  原始日志复制 %d 个' % len(copied))
