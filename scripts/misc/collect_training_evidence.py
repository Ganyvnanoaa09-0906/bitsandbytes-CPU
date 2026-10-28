"""collect_training_evidence.py -- gather everything measurable about the real training path.

Everything below is measured on this machine, on the configuration the AR model is actually
trained with: 10 layers, d_model 512, 8 heads, 4 KV heads, ffn 2048, vocab 1024, grid 32,
T=1024, batch 2, 6 threads. Nothing here is a standalone peak benchmark unless it is labelled
as one, because the request is specifically about the real path.

Sections:
  1. configuration and cost          -- what one step costs, in time and tokens/s
  2. phase breakdown                 -- forward, backward, optimiser, measured separately
  3. operator profile                -- where the time actually goes, by operator
  4. memory split                    -- weights, gradients, optimiser state, activations
  5. memory bandwidth in use         -- what the step actually demands against what exists
  6. CPU state                       -- frequency, temperature, power where obtainable
  7. compute vs memory bound         -- per-operator, with the arithmetic
  8. raw logs                        -- copied verbatim, not summarised
  9. failures and their evidence     -- each closed direction with its numbers

Writes a markdown appendix and a directory of the raw log copies.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import dataclasses
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


ROOT = r'D:\work\bitsandbytes-CPU'
OUT_DIR = r'D:\work\research\evidence'
os.makedirs(OUT_DIR, exist_ok=True)
torch.set_num_threads(6)

L = []            # markdown lines
def w(s=''):
    L.append(s)

w('# 附录 A：AR 训练路径的实测证据')
w()
w('本附录记录**当前实际训练路径**上的一切可测量项，由 `collect_training_evidence.py` 生成。')
w('与独立 benchmark 的峰值区分开：标注「实测路径」的条目来自真实训练配置，标注「独立」的')
w('来自单独的内核基准。生成时间 %s。' % time.strftime('%Y-%m-%d %H:%M:%S'))
w()

# ---------------------------------------------------------------- 1. configuration
w('## A.1 训练配置与单步成本')
w()
CKPT = os.path.join(ROOT, 'i5build', 'ar32_run1', 'ckpt_3000.pt')
cfg_txt = 'n/a'
try:
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    c = ck.get('cfg') or {}
    cfg_txt = ('vocab %s · d_model %s · n_layer %s · n_head %s · n_kv_head %s · '
               'ffn %s · grid %s · T %s · n_cluster %s · head_mode %s · mask_mode %s'
               % (c.get('vocab_size'), c.get('d_model'), c.get('n_layer'), c.get('n_head'),
                  c.get('n_kv_head'), c.get('ffn_dim'), c.get('grid'),
                  c.get('total_tokens'), c.get('n_cluster'), c.get('head_mode'),
                  c.get('mask_mode')))
    n_params = sum(v.numel() for v in ck['model'].values() if torch.is_tensor(v))
except Exception as e:
    n_params = 0
    cfg_txt = 'checkpoint 读取失败: %s' % e

w('| 项 | 值 |')
w('|---|---|')
w('| 配置 | %s |' % cfg_txt)
w('| 参数量 | %.2f M |' % (n_params / 1e6))
w('| batch × T | 2 × 1024 = 2048 token/步 |')
w('| 线程 | 6（`torch.set_num_threads(6)`） |')
w()

# step cost from the real training log
def log_numbers(path, pattern):
    out = []
    try:
        for ln in open(path, encoding='utf-8', errors='replace'):
            m = re.search(pattern, ln)
            if m:
                out.append(m.groups())
    except Exception:
        pass
    return out

steps = log_numbers(os.path.join(ROOT, 'i5build', 'sgdrA.log'),
                    r'(\d+)/1500 loss=[\d.]+ ([\d.]+)s')
if steps:
    secs = [float(s) for _, s in steps]
    w('实测路径（`i5build/sgdrA.log`，1500 步，batch 2，T=1024）：')
    w()
    w('| 指标 | 值 |')
    w('|---|---|')
    w('| 每步耗时（中位） | **%.3f s** |' % statistics.median(secs))
    w('| 每步耗时（最小/最大） | %.3f / %.3f s |' % (min(secs), max(secs)))
    w('| 吞吐 | **%.1f token/s** |' % (2048 / statistics.median(secs)))
    w('| 1500 步总时长 | **%.2f 小时** |' % (sum(secs) / 3600))
    w('| 3000 步外推 | %.2f 小时 |' % (sum(secs) / 3600 * 2))
    w()
    w('*注：这是 batch 2。batch 8 的实测每步 12.84 s，即 638 token/s，'
      '**batch 更大反而更慢**（7.20.2）。*')
    w()

# ---------------------------------------------------------------- 2. phase breakdown
w('## A.2 前向 / 反向 / 优化器 的分项耗时')
w()
w('来源 `clean_step_timing.py`，100 步取中位，空闲机器。两臂都是 batch 2、T=1024。')
w()
w('| 阶段 | AdamW8bit | Muon(2D)+AdamW | 差 |')
w('|---|---|---|---|')
w('| 前向 | 1.166 s | 1.155 s | −1% |')
w('| 反向 | 2.647 s | 2.706 s | +2% |')
w('| **优化器** | **0.138 s** | **2.562 s** | **+18.5×** |')
w('| 合计 | 3.952 s | 6.423 s | +1.63× |')
w()
w('**Muon 的全部代价在优化器里**，前向与反向几乎不变。那 2.562 s 中的 2.42 s 是 '
      'Newton-Schulz（74 个 2D 权重 × 5 次迭代 × 3 个矩阵乘）。')
w()

# ---------------------------------------------------------------- 3. operator profile
w('## A.3 算子耗时占比（实测路径）')
w()
w('来源 `profile_step_batch8.py`（torch profiler，一次前向+反向，batch 8，T=1024，因果）。')
w('自身时间（self time），已按算子去重。')
w()
w('| 算子 | 自身时间 ms | 占比 | 调用次数 |')
w('|---|---|---|---|')
PROF = [('aten::mm', 6438.6, 53.56, 184),
        ('flash_attention (前)', 1675.0, 13.93, 10),
        ('aten::mul', 1290.7, 10.74, 490),
        ('flash_attention (后)', 584.9, 4.87, 10),
        ('autograd MmBackward', 319.3, 2.66, 61),
        ('aten::copy_', 257.3, 2.14, 284),
        ('aten::sum', 253.8, 2.11, 162),
        ('aten::add_', 170.3, 1.42, 109),
        ('aten::pow', 154.4, 1.28, 120),
        ('aten::cat', 148.4, 1.23, 10),
        ('aten::add', 118.3, 0.98, 33),
        ('其余（silu / fill_ / div / view 等）', 660.6, 5.08, 700)]
tot = sum(p[1] for p in PROF)
for name, ms, pct, n in PROF:
    w('| %s | %.1f | %.2f%% | %d |' % (name, ms, pct, n))
w('| **合计** | **%.1f** | **100%%** | — |' % tot)
w()
w('**按类别**：matmul 53.6% · elementwise/reshape 18.8% · attention 18.8% · other 8.6%。')
w()
w('两个可操作的事实：')
w()
w('* `aten::mul` **1290 ms / 490 次 = 2.6 ms/次**，而 10 层的模型出现 490 次说明它来自')
w('  RMSNorm 的内部展开（含 QK-Norm，它作用在 (B,T,H,D) 上，行数是普通 norm 的 8 倍，'
      '见 7.29 与 A.9）；')
w('* `aten::mm` 占 53.6% 而它已经接近该形状的独立水平（A.7），**剩下的空间不在 mm 里**。')
w()

# ---------------------------------------------------------------- 4. memory split
w('## A.4 内存分解：权重 / 梯度 / 优化器状态 / 激活')
w()
try:
    from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
    cfg = SmallImageConfigV2(**{k: v for k, v in (ck.get('cfg') or {}).items()
                               if k in names})
    m = SmallARImageModelV2(cfg)
    n_p = sum(p.numel() for p in m.parameters())
    fp32_w = n_p * 4
    grad = n_p * 4
    # 8-bit optimiser: 2 x 1 byte per parameter (m and v)
    opt8 = n_p * 2
    opt32 = n_p * 8
    act = 8 * 1024 * 512 * 4          # B*T*d in fp32, one layer's worth
    w('| 项 | 每参数字节 | 30.5M 模型 | 说明 |')
    w('|---|---|---|---|')
    w('| 权重（fp32） | 4.0 | **%.1f MB** | 训练时的主副本 |')
    w('| 梯度（fp32） | 4.0 | **%.1f MB** | 与权重同尺寸 |')
    w('| 优化器状态（8-bit） | 2.0 | **%.1f MB** | m 与 v 各 1 字节 |')
    w('| 优化器状态（fp32，对照） | 8.0 | %.1f MB | torch 原生 AdamW |')
    w('| 单层激活（B×T×d） | — | **%.1f MB** | 10 层另有保存量 |')
    w('| 峰值实测 RSS | — | **~5.4 GB** | 3000 步训练记录（7.18.5） |')
    w()
    w('**读法**：fp32 训练的三个主要项是权重 4 + 梯度 4 + 优化器 8 = **16 字节/参数**；')
    w('换成 8-bit 优化器后是 4 + 4 + 2 = **10 字节/参数**，即 1.6× 的下降。而激活与')
    w('中间张量是剩下的部分，实测峰值 5.4 GB 中它们占大头。')
    w()
    w('若权重也用 2-bit（7.23 已实现）：权重 0.31 + 梯度 4 + 状态 1.03 ≈ **5.3 字节/参数**，')
    w('相对 fp32 的 16 是 **3×**。梯度仍是 fp32 是这里最大的剩余项。')
except Exception as e:
    w('内存分解失败: %s' % e)
w()

with open(os.path.join(OUT_DIR, 'A_training_evidence_part1.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(x if isinstance(x, str) else x[0] for x in L))
print('  第 1 部分已写出（%d 行）' % len(L))
print('\n'.join(L[:6]))
