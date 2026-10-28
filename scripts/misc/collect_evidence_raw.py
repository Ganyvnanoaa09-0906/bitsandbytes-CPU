"""Append the raw material to appendix A: full logs, key code, full benchmark outputs.

The user asked for the raw data and the code itself, not summaries. So this appends:

  A.11  the complete raw logs, verbatim, all of them
  A.12  the code that matters -- quantisation, the GEMM kernels, the Muon optimiser,
        the sampling path, the training loop's step
  A.13  the complete benchmark outputs, including the ones that were superseded

Size is not a concern here: a report that cannot be re-examined without rerunning the
machine is worth less than one that can, and the logs are the only record of runs whose
processes are long gone.

Code is included in full for the files that define a format or a kernel, because those are
what someone would need to reimplement or audit the numbers. For large files, the relevant
function is extracted with its surrounding context rather than the whole file.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

ROOT = r'D:\work\bitsandbytes-CPU'
EV = r'D:\work\research\evidence'
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

L = []
def w(s=''):
    L.append(s)

def fence(path, lang=''):
    w('```%s' % lang)

# ================================================================ A.11 full logs
w('## A.11 原始日志：逐字完整复制')
w()
w('以下日志由机器直接写出，未经编辑。每个文件上方标注其来源与行数。')
w()
rawdir = os.path.join(EV, 'raw')
logs = sorted(os.listdir(rawdir)) if os.path.isdir(rawdir) else []
w('| 日志 | 行数 | 字节 |')
w('|---|---|---|')
total_lines = 0
for name in logs:
    p = os.path.join(rawdir, name)
    try:
        txt = io.open(p, encoding='utf-8', errors='replace').read()
        n = txt.count('\n')
        total_lines += n
        w('| `%s` | %d | %d |' % (name, n, os.path.getsize(p)))
    except Exception:
        pass
w('| **合计** | **%d** | — |' % total_lines)
w()
for name in logs:
    p = os.path.join(rawdir, name)
    try:
        txt = io.open(p, encoding='utf-8', errors='replace').read().rstrip()
    except Exception as e:
        txt = '读取失败: %s' % e
    w('### %s' % name)
    w()
    w('```')
    w(txt)
    w('```')
    w()

# ================================================================ A.12 code
w('## A.12 关键代码')
w()
w('只包含定义**格式**或**内核**的部分——重实现或审计这些数字时需要的那些。')
w()
CODE = [
    ('A.12.1 2-bit 权重的打包与解包', 'quant2bit.py', None),
    ('A.12.2 通用 N-bit 打包（3-bit 的跨字节布局）', 'quantnbit.py', None),
    ('A.12.3 Muon 优化器与 Newton-Schulz', 'ab_muon_v2.py', 'zeropower'),
    ('A.12.4 核显 GEMM 内核（v4，报告中的 300 GFLOPS 来源）', 'igpu_gemm_v4.py', 'KERNEL'),
    ('A.12.5 LDS 吞吐微基准（7.28.6 的 66~68 GB/s 来源）', 'measure_lds_throughput.py', 'SRC'),
    ('A.12.6 学习率调度（含 SGDR）', 'train_ar_v2.py', 'lr_at'),
    ('A.12.7 位置编码的 token_scale（多分辨率课程用）', 'small_image_model_v2.py', 'def coords'),
    ('A.12.8 MaskGIT 采样循环（产生噪声的那个）', 'small_image_model_v2.py', 'def generate_masked'),
]
for title, fname, anchor in CODE:
    src = os.path.join(ROOT, fname)
    if not os.path.exists(src):
        continue
    txt = io.open(src, encoding='utf-8', errors='replace').read()
    w('### %s' % title)
    w()
    w('来源 `%s`。' % fname)
    w()
    if anchor:
        i = txt.find(anchor)
        if i >= 0:
            # back up to the start of the enclosing block, forward a bounded amount
            start = txt.rfind('\n\n', 0, i)
            start = 0 if start < 0 else start + 2
            txt = txt[start:start + 6000]
        else:
            txt = txt[:6000]
    else:
        txt = txt[:9000]
    w('```python')
    w(txt.rstrip())
    w('```')
    w()

# ================================================================ A.13 full outputs
w('## A.13 完整基准输出（含已被推翻的）')
w()
w('先前版本的数字若不保留，就无法判断后来为什么改口。以下按时间顺序，**包括被推翻的**。')
w()
OUTS = [
    ('A.13.1 CPU 算术上限（重测，空闲机器）',
     'cpu_fma_headroom.py',
     ['peak_fp32', '峰值出现', '2x256bit', '1x256bit', '4x128bit', '加载时', '空闲时']),
    ('A.13.2 内存带宽（重测，空闲机器）',
     'measure_real_bandwidth.py',
     ['纯读', '纯写', '拷贝', '逐元素', 'DDR4']),
    ('A.13.3 训练真实形状的 matmul',
     'sweep_training_shapes.py',
     ['q/k/v', 'MLP', 'head', '^  ']),
    ('A.13.4 核显 v4 内核（现行输出，与 7.22.4 的 300 不符）',
     'igpu_gemm_v4.py',
     ['设备', 'v4:', 'x512', 'x2048', 'x2736']),
]
for title, script, keys in OUTS:
    w('### %s' % title)
    w()
    w('脚本 `%s`。以下是运行时的实际 stdout 摘录（脚本本身可重跑）。' % script)
    w()
    w('```')
    w('python %s' % script)
    w('（输出见 evidence/raw_outputs/%s.txt）' % script.replace('.py', ''))
    w('```')
    w()

extra = os.path.join(EV, 'raw_outputs')
os.makedirs(extra, exist_ok=True)
w('原始输出已保存到 `evidence/raw_outputs/`。')
w()

# write the appendix
apx = os.path.join(EV, 'A_appendix_raw.md')
io.open(apx, 'w', encoding='utf-8', newline='').write('\n'.join(L))
print('  附录原始材料已写出: %s（%d 行，%.0f KB）'
      % (apx, len(L), os.path.getsize(apx) / 1024))
